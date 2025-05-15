# coding=utf-8


import argparse
import glob
import logging
import os
import timeit
import math
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random

import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import copy
from scipy.linalg import svd
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLMRobertaForMaskedLM,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    XLMRobertaTokenizer
)
from transformers import AutoModel


from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from xlm_roberta import XLMRobertaForQuestionAnswering, XLMRobertaConfig
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
from processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "xlm-roberta": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer)
}

MODEL_MASK_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "xlm-roberta": (XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def qa_loss(start_logits, end_logits, start_positions, end_positions, return_start_end=False):
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2

    if not return_start_end:
        return total_loss
    else:
        return total_loss, start_loss, end_loss


def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def METHOD(args, model, tokenizer, model_initial_state, option_max, prefix="", language='en', lang2id=None):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True,
                                                          language=language, lang2id=lang2id)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    history_pc = None

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()
    teacher_model_path = os.path.join(args.output_dir, f"teacher_final_checkpoint_{args.option - 1}")
    print(teacher_model_path)
    if args.option == 1:
        teacher_model = copy.deepcopy(model)
    else:
        teacher_model = copy.deepcopy(model)
        teacher_model_temp = XLMRobertaForQuestionAnswering.from_pretrained(teacher_model_path)
        teacher_model.load_state_dict(teacher_model_temp.state_dict())

    logger.info("teacher mode is teacher_final_checkpoint_{}.".format(args.option - 1))
    teacher_model.eval()

    initial_model = copy.deepcopy(model)
    initial_model.load_state_dict(model_initial_state)

    teacher_model_device = next(teacher_model.parameters()).device
    model_initial_state = {k: v.to(teacher_model_device) for k, v in model_initial_state.items()}

    global_step = 0

    memory_size = args.memory_size
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    train_batchs = [d for d in epoch_iterator]

    for step, train_batch in enumerate(train_batchs):
        ## do test
        model.eval()
        train_batch = tuple(t.to(args.device) for t in train_batch)

        with torch.no_grad():
            train_inputs = {
                "input_ids": train_batch[0],
                "attention_mask": train_batch[1],
            }
            example_indices = train_batch[3]

            outputs = model(**train_inputs)

            # debias
            if args.debias and args.alpha != 1:
                with torch.no_grad():
                    source_outputs = teacher_model(**train_inputs)
                    pseudo_start_logits = source_outputs[0]
                    pseudo_end_logits = source_outputs[1]
                    beta = args.alpha
                    start_logits = 2 * outputs[0] - pseudo_start_logits - beta * (outputs[0] - pseudo_start_logits)
                    end_logits = 2 * outputs[1] - pseudo_end_logits - beta * (outputs[1] - pseudo_end_logits)
                    outputs = (start_logits, end_logits)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

        ## do adaptation
        if step < memory_size:
            pool_size = step + 1
        else:
            pool_size = memory_size

        for train_batch in train_batchs[step + 1 - pool_size: step + 1]:
            model.train()
            train_batch = tuple(t.to(args.device) for t in train_batch)

            train_inputs = {
                "input_ids": train_batch[0],
                "attention_mask": train_batch[1],
            }

            if args.m != 0:
                if args.topk > 0:
                    # OIL
                    alpha = args.m
                    with torch.no_grad():
                        source_outputs = teacher_model(**train_inputs)
                        pseudo_start_logits = source_outputs[0].detach()
                        pseudo_end_logits = source_outputs[1].detach()
                        pseudo_start_positions = torch.argmax(pseudo_start_logits, dim=1)
                        pseudo_end_positions = torch.argmax(pseudo_end_logits, dim=1)
                    adapted_outputs = model(**train_inputs)
                    start_logits = adapted_outputs[0]
                    end_logits = adapted_outputs[1]

                    if not args.debias:
                        _, loss_hard_start, loss_hard_end = qa_loss(start_logits, end_logits,
                                                                    copy.deepcopy(pseudo_start_positions),
                                                                    copy.deepcopy(pseudo_end_positions),
                                                                    return_start_end=True)
                        echo = args.topk
                        loss = (torch.sum((loss_hard_start < echo) * loss_hard_start) / (
                                torch.sum((loss_hard_start < echo)) + 1e-10) + \
                                torch.sum((loss_hard_end < echo) * loss_hard_end) / (
                                        torch.sum((loss_hard_end < echo)) + 1e-10)) / 2.

                    elif args.debias:
                        _, loss_hard_start, loss_hard_end = qa_loss(start_logits, end_logits,
                                                                    copy.deepcopy(pseudo_start_positions),
                                                                    copy.deepcopy(pseudo_end_positions),
                                                                    return_start_end=True)
                        start_logits_2 = (start_logits + start_logits - pseudo_start_logits)
                        end_logits_2 = (end_logits + end_logits - pseudo_end_logits)
                        _, loss_clean_start, loss_clean_end = qa_loss(start_logits_2, end_logits_2,
                                                                      copy.deepcopy(pseudo_start_positions),
                                                                      copy.deepcopy(pseudo_end_positions),
                                                                      return_start_end=True)
                        echo = args.topk
                        loss = (torch.sum((loss_hard_start < echo) * loss_clean_start) / (
                                torch.sum((loss_hard_start < echo)) + 1e-10) + \
                                torch.sum((loss_hard_end < echo) * loss_clean_end) / (
                                        torch.sum((loss_hard_end < echo)) + 1e-10)) / 2.

                elif args.topk == 0:
                    alpha = args.m

                    # cotta
                    def data_augmentation(train_inputs, replace_prob=0.1, delete_prob=0.1, vocab_size=10000,
                                          pad_token_id=0):
                        augmented_inputs = {}
                        device = train_inputs['input_ids'].device
                        for key in train_inputs.keys():
                            if key == 'input_ids':
                                replace_mask = torch.rand_like(train_inputs[key], dtype=torch.float32) < replace_prob
                                delete_mask = torch.rand_like(train_inputs[key], dtype=torch.float32) < delete_prob
                                conflict_mask = replace_mask & delete_mask
                                replace_mask = replace_mask & ~conflict_mask
                                delete_mask = delete_mask & ~conflict_mask
                                random_values = torch.randint(0, vocab_size, size=train_inputs[key].shape,
                                                              device=device)
                                augmented_inputs[key] = torch.where(replace_mask, random_values, train_inputs[key])
                                augmented_inputs[key] = torch.where(delete_mask,
                                                                    torch.full_like(train_inputs[key], pad_token_id,
                                                                                    device=device),
                                                                    augmented_inputs[key])
                            else:
                                augmented_inputs[key] = train_inputs[key]

                        return augmented_inputs

                    with torch.no_grad():
                        teacher_outputs = teacher_model(**train_inputs)
                        teacher_start_logits = teacher_outputs[0].detach()
                        teacher_end_logits = teacher_outputs[1].detach()
                        teacher_start_probs = F.softmax(teacher_start_logits, dim=1)
                        teacher_end_probs = F.softmax(teacher_end_logits, dim=1)

                    max_start_values, _ = torch.max(teacher_start_probs, dim=1)
                    max_end_values, _ = torch.max(teacher_end_probs, dim=1)
                    avg_max_start_value = max_start_values.detach().cpu().numpy().mean()
                    avg_max_end_value = max_end_values.detach().cpu().numpy().mean()
                    if avg_max_start_value > 0.69:
                        final_teacher_start_position = torch.argmax(teacher_start_probs, dim=1)
                    else:
                        augmented_start_logits = []
                        for _ in range(32):
                            augmented_inputs = data_augmentation(train_inputs)
                            with torch.no_grad():
                                augmented_outputs = teacher_model(**augmented_inputs)
                            augmented_start_logits.append(augmented_outputs[0].detach())
                        avg_augmented_start_logits = torch.stack(augmented_start_logits, dim=0).mean(0)
                        final_teacher_start_position = torch.argmax(avg_augmented_start_logits, dim=-1)

                    if avg_max_end_value > 0.69:
                        final_teacher_end_position = torch.argmax(teacher_end_probs, dim=1)
                    else:
                        augmented_end_logits = []
                        for _ in range(32):
                            augmented_inputs = data_augmentation(train_inputs)
                            with torch.no_grad():
                                augmented_outputs = teacher_model(**augmented_inputs)
                            augmented_end_logits.append(augmented_outputs[1].detach())
                        avg_augmented_end_logits = torch.stack(augmented_end_logits, dim=0).mean(0)
                        final_teacher_end_position = torch.argmax(avg_augmented_end_logits, dim=-1)

                    adapted_outputs = model(**train_inputs)
                    student_start_logits = adapted_outputs[0]
                    student_end_logits = adapted_outputs[1]
                    loss = qa_loss(student_start_logits, student_end_logits,
                                   final_teacher_start_position,
                                   final_teacher_end_position).mean(0)

                elif args.topk == -1:
                    # CTTA-T
                    def get_teacher_position_with_dropout(teacher_logits, teacher_model, inputs,
                                                          if_start,
                                                          dropout_num=8, threshold=args.threshold,
                                                          consistency_threshold=args.contribution_threshold):

                        with torch.no_grad():
                            teacher_probs = F.softmax(teacher_logits, dim=1)
                            _, teacher_max_position = teacher_probs.max(dim=1)
                            entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-12), dim=1)
                        entropy_thresh = threshold * math.log(384)
                        need_mc =(entropy > entropy_thresh)
                        dropout_list = torch.where(need_mc)[0].tolist()
                        low_consistency_idx = []

                        if len(dropout_list) > 0:
                            original_mode = teacher_model.training
                            teacher_model.train()

                            try:
                                batch_inputs = {
                                    "input_ids": inputs["input_ids"][dropout_list],
                                    "attention_mask": inputs["attention_mask"][dropout_list]
                                }
                                batch_size = len(dropout_list)
                                dropout_samples_list = []
                                for _ in range(dropout_num):
                                    with torch.no_grad():
                                        if if_start:
                                            logits = teacher_model(**batch_inputs)[0]
                                        else:
                                            logits = teacher_model(**batch_inputs)[1]
                                        sampled_probs = F.softmax(logits, dim=1)
                                        dropout_samples_list.append(sampled_probs)

                                confidence_matrix = torch.stack(dropout_samples_list, dim=0).cpu().numpy()

                                for idx, sample_idx in enumerate(dropout_list):
                                    sample_cm = confidence_matrix[:, idx, :]
                                    u, s, vt = np.linalg.svd(sample_cm, full_matrices=False)

                                    s_tensor = torch.from_numpy(s).to(teacher_probs.device).float()
                                    vt_tensor = torch.from_numpy(vt).to(teacher_probs.device).float()
                                    consistency = torch.norm(
                                        vt_tensor * s_tensor.reshape(-1, 1),
                                        p=2,
                                        dim=0
                                    )

                                    consistency_score = torch.max(consistency)
                                    combined_probs = teacher_probs[sample_idx] * F.softmax(consistency, dim=0)
                                    predicted_label = torch.argmax(combined_probs)
                                    combined_probs = F.softmax(combined_probs, dim=0)
                                    teacher_max_position[sample_idx] = predicted_label

                                    if consistency_score < consistency_threshold:
                                        low_consistency_idx.append(sample_idx)
                            finally:
                                teacher_model.train(original_mode)

                        return teacher_max_position, low_consistency_idx

                    adapted_outputs = model(**train_inputs)
                    student_start_logits = adapted_outputs[0]
                    student_end_logits = adapted_outputs[1]
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**train_inputs)
                        teacher_start_logits = teacher_outputs[0].detach()
                        teacher_end_logits = teacher_outputs[1].detach()

                        # # IPCA update ALPHA
                        input_ids = train_inputs["input_ids"]
                        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
                        position_ids = position_ids.unsqueeze(0).expand(input_ids.size())
                        word_embed = model.roberta.embeddings.word_embeddings(input_ids)
                        pos_embed = model.roberta.embeddings.position_embeddings(position_ids)
                        embeddings_1 = word_embed + pos_embed
                        embeddings_all = embeddings_1.reshape(-1, embeddings_1.shape[-1])
                        num_tokens = embeddings_all.shape[0]
                        sample_size = min(384, num_tokens)
                        sample_indices = np.random.choice(num_tokens, size=sample_size, replace=False)
                        embeddings = embeddings_all[sample_indices].detach().cpu().numpy()

                        k = 3
                        if step == 0:
                            ipca = IncrementalPCA(n_components=k)
                            ipca.fit(embeddings)
                            history_pc = ipca.components_
                            alpha = args.m
                        else:
                            ipca.partial_fit(embeddings)
                            new_pc = ipca.components_
                            similarity = cosine_similarity(new_pc, history_pc).mean()
                            alpha_bound = args.m
                            mapped_similarity = ((1 - alpha_bound) / 2) * similarity + (alpha_bound + 1) / 2
                            alpha = mapped_similarity
                            history_pc = ipca.components_

                        teacher_start_position, lc_start_idx = get_teacher_position_with_dropout(
                                                                teacher_start_logits,
                                                                teacher_model,
                                                                train_inputs,
                                                                True
                                                                )
                        teacher_end_position, lc_end_idx = get_teacher_position_with_dropout(
                                                            teacher_end_logits,
                                                            teacher_model,
                                                            train_inputs,
                                                            False
                                                            )
                    # loss
                    low_consistency_idx = list(set(lc_start_idx + lc_end_idx))
                    mask = torch.ones_like(teacher_start_position, dtype=torch.bool)
                    mask[low_consistency_idx] = False
                    loss = qa_loss(student_start_logits[mask, :],
                                   student_end_logits[mask, :],
                                   teacher_start_position[mask],
                                   teacher_end_position[mask]
                                   ).mean(0)


            elif args.m == 0 and args.topk > 0:
                # PL
                alpha = args.m
                with torch.no_grad():
                    source_outputs = model(**train_inputs)
                    pseudo_start_logits = source_outputs[0].detach()
                    pseudo_end_logits = source_outputs[1].detach()
                    pseudo_start_positions = torch.argmax(pseudo_start_logits, dim=1)
                    pseudo_end_positions = torch.argmax(pseudo_end_logits, dim=1)

                adapted_outputs = model(**train_inputs)
                start_logits = adapted_outputs[0]
                end_logits = adapted_outputs[1]

                loss_hard = qa_loss(start_logits, end_logits, pseudo_start_positions, pseudo_end_positions)
                echo = args.topk
                loss = torch.sum((loss_hard < echo) * loss_hard) / (torch.sum((loss_hard < echo)) + 1e-10)

            elif args.m == 0 and args.topk == -1:
                # Tent
                alpha = args.m
                with torch.no_grad():
                    source_outputs = teacher_model(**train_inputs)
                    pseudo_start_logits = source_outputs[0].detach()
                    pseudo_end_logits = source_outputs[1].detach()
                adapted_outputs = model(**train_inputs)
                start_logits = adapted_outputs[0]
                end_logits = adapted_outputs[1]

                start_loss = -(F.softmax(pseudo_start_logits, dim=1) * F.log_softmax(start_logits, dim=1)).sum(1).mean(
                    0)
                end_loss = -(F.softmax(pseudo_end_logits, dim=1) * F.log_softmax(end_logits, dim=1)).sum(1).mean(0)
                loss = (start_loss + end_loss) / 2


            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            ## update the source model


            for param1, param2 in zip(teacher_model.parameters(), model.parameters()):
                param1.data = alpha * param1.data + (1. - alpha) * param2.data

            if args.ccc_restore:
                for name, param in teacher_model.named_parameters():
                    if 'weight' in name or 'bias' in name:
                        mask = (torch.rand_like(param) < 0.01).float()
                        mask = mask.to(param.device)
                        key = name
                        if key in model_initial_state:
                            initial_value = model_initial_state[key].data.to(param.device)
                            with torch.no_grad():
                                param.data = torch.where(mask == 1, initial_value, param.data)

            if args.cotta_restore:
                for nm, m in model.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape) < 0.01).float().cuda()
                            with torch.no_grad():
                                p.data = model_initial_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

        global_step += 1

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    # save student/teacher model
    if args.option <= option_max:
        teacher_model_output_path = os.path.join(args.output_dir, f"teacher_final_checkpoint_{args.option}")
        save_model(teacher_model, tokenizer, teacher_model_output_path)
        save_model(model, tokenizer, os.path.join(args.output_dir, "checkpoint_last"))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}_{}.json".format(language, prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}.json".format(language, prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
            map_to_origin=not (args.model_type == "xlm-roberta" and (language == 'zh' or language == "ko")),
        )

    # Compute the F1 and exact scores.
    print('\n\n\n')
    print('\t\t Results on %s' % (language))
    results = squad_evaluate(examples, predictions)
    result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in results.items())
    result["language"] = language
    logger.info("Results: {}".format(result))
    print('\n\n\n')
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False,
                            language='en', lang2id=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            os.path.basename(args.predict_file) if evaluate else os.path.basename(args.train_file),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(language)
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate, language=language)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file, language=language)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
            lang2id=lang2id
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--initial_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--memory_size",
        type=int,
        default=5
    )

    parser.add_argument(
        "--ccc_restore",
        type=int,
        default=0
    )

    parser.add_argument(
        "--cotta_restore",
        type=int,
        default=0
    )

    parser.add_argument(
        "--topk",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--debias",
        action="store_true"
    )

    parser.add_argument(
        "--do_METHOD",
        action="store_true"
    )

    parser.add_argument(
        "--alpha",
        default=1,
        type=float
    )

    parser.add_argument(
        "--m",
        default=0.99,
        type=float
    )
    parser.add_argument(
        "--option",
        default=1,
        type=int
    )

    parser.add_argument(
        "--option_max",
        default=1,
        type=int
    )
    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float
    )
    parser.add_argument(
        "--contribution_threshold",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--train_lang", type=str, default="en", help="The language of the training data")
    parser.add_argument("--eval_lang", type=str, default="en", help="The language of the test data")

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    set_seed(args)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.model_type == "xlm":
        config.use_lang_emb = True
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("lang2id = {}".format(lang2id))

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    option_max = args.option_max
    results = {}
    all_res = []
    if args.do_METHOD and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)
            model_initial = XLMRobertaForQuestionAnswering.from_pretrained(args.initial_model_path)
            model_initial_state = copy.deepcopy(model_initial.state_dict())
            if args.do_METHOD:
                result = METHOD(args, model, tokenizer, model_initial_state, option_max, prefix=global_step,
                                language=args.eval_lang, lang2id=lang2id)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

            all_res.append(result)

    return results


if __name__ == "__main__":
    main()
