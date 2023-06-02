# To control logging level for various modules used in the application:
import argparse
import datetime
import logging
import math
import os
import pickle
import random
import re
import warnings
from collections import Counter
import numpy as np
import torch
from datasets import load_from_disk
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, PrinterCallback, Trainer,
                          TrainingArguments)
from transformers.integrations import TensorBoardCallback
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


def set_global_logging_level(level, prefix):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefix) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def set_random_seeds(random_seed):
    """
    Sets all possible random seeds so results can be reproduced
    """
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def compute_metrics(eval_pred):
    global num_class, id2label
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1_all_list = f1_score(
        y_pred=predictions,
        y_true=labels,
        labels=list(range(num_class)),
        average=None,
        zero_division=0,
    )
    result = dict()
    result.update(
        {
            "f1_micro": f1_score(
                y_pred=predictions,
                y_true=labels,
                labels=list(range(num_class)),
                average="micro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_pred=predictions,
                y_true=labels,
                labels=list(range(num_class)),
                average="macro",
                zero_division=0,
            ),
            "accuracy": accuracy_score(y_true=labels, y_pred=predictions),
            "confusion_matrix": confusion_matrix(
                y_true=labels, y_pred=predictions, labels=list(range(num_class))
            ),
        }
    )
    for lbl_id in range(num_class):
        result.update({f"f1_{id2label[lbl_id]}": f1_all_list[lbl_id]})
    return result


def checkflip(p):
    random_value = random.random()
    flag = False
    if random_value < p:
        flag = True
    return flag, random_value


def flip(value, flag, num_class):
    if not flag or value == num_class:
        return value
    start = 0
    end = num_class
    return random.choice([k for k in range(start, end) if k != value])


def get_tls_from_memory(time_last_seen_array, cur_index, mem_cnt):
    time_last_seen = dict()
    for lbl, tls_mem in time_last_seen_array.items():
        tls_mem_relative = [cur_index - x for x in tls_mem if x != -1]
        non_zero_tls_mem = max(len([_ for _ in tls_mem if _ != -1]), 1)
        tls_avg = (
            sum(tls_mem_relative) / non_zero_tls_mem
        )
        time_last_seen[lbl] = tls_avg
    return time_last_seen


warnings.simplefilter(action="ignore", category=FutureWarning)
set_global_logging_level(level=logging.ERROR, prefix=[""])
set_verbosity_error()
disable_progress_bar()

num_class = 5
id2label = {0: "sadness", 1: "joy", 2: "surprise", 3: "anger", 4: "fear"}


def do_testing(args):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    def update_label(examples, idx):
        examples["label"] = selected_indices_label[idx]
        return examples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_cpu:
        print("Forcing to CPU only inference")
        device = torch.device("cpu")
    print(device)

    # Setting up data for AL training
    if args.data_type == "emotion":
        global num_class, id2label
        num_class = 5
        id2label = {
            0: "sadness",
            1: "joy",
            2: "surprise",
            3: "anger",
            4: "fear",
        }
        label2id = {y: x for x, y in id2label.items()}

    model_name = args.al_bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_split = load_from_disk(args.data_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_split_w_tokens = data_split.map(preprocess_function, batched=True)
    # Finished setting up data

    # Human simulation parameters set
    decay_params = [float(x) for x in args.mem_decay.split(",")]  # [0.3,6,1]
    mem_cnt = args.tls_memory_size  # 3
    alpha, beta, delta = decay_params

    # Active learning parameters set
    budget = args.budget
    train_every_x = args.train_every_x
    random_seed_values = [int(x) for x in args.seed_values.split(",")]
    al_sampling_type = args.al_sampling_type
    print(f"{'='*50}\nAL Sampling: {al_sampling_type}\n")

    # Variables to store results
    now = datetime.datetime.now()
    score_eval_loss = []
    score_eval_f1_micro = []
    score_eval_f1_macro = []
    score_eval_f1_sadness = []
    score_eval_f1_joy = []
    score_eval_f1_anger = []
    score_eval_f1_fear = []
    score_eval_f1_surprise = []
    score_train_explore = []
    score_error_lbl_score = []
    score_train_text = []
    score_train_label = []
    score_error_lbl_values = []


    training_args = TrainingArguments(
        output_dir="_tmp_model",
        learning_rate=2e-5,  # fixed
        per_device_train_batch_size=8,  # fixed
        per_device_eval_batch_size=16,  # fixed
        num_train_epochs=5,  # 5,  # 5,  # fixed
        weight_decay=0.01,  # fixed
        logging_strategy="no",
        evaluation_strategy="no",
        save_strategy="no",
        push_to_hub=False,
    )

    for sd in random_seed_values:
        print(f"For Random Seed: {sd}")
        set_random_seeds(sd)
        train_dataset_all = data_split_w_tokens["train"].shuffle(seed=sd)
        time_last_seen_array = {x: [-1] * mem_cnt for x in range(num_class)}
        time_last_seen = get_tls_from_memory(time_last_seen_array, 0, mem_cnt)
        tls = [time_last_seen[lbl] for lbl in range(num_class)]
        err_lbl_score = 0
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_class, id2label=id2label, label2id=label2id
        )
        model.to(device)
        for name, param in model.named_parameters():
            if name.startswith("bert.embeddings"):
                param.requires_grad = False

        selected_indices = []
        selected_indices_label = []
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=data_split_w_tokens["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.remove_callback(TensorBoardCallback)
        trainer.remove_callback(PrinterCallback)

        cur_idx = -1
        assert al_sampling_type in [
            "diversity",
        ], f"Incorrect sampling strategy Choose diversity"
        assert args.mem_decay_type in ["sigmoid", "exponential"], f"Only sigmoid and exponential decays are available"
        print("Performing Clustering")
        clustering = AgglomerativeClustering(n_clusters=budget)
        cluster_feats = np.array(train_dataset_all["feat"])
        clustering.fit(cluster_feats)
        cluster_centers = []
        for i in range(budget):
            cluster_centers.append(np.mean(cluster_feats[clustering.labels_ == i], axis=0))
        selected_diversity_distances = pairwise_distances(cluster_centers, cluster_feats)
        selected_diversity_indices = []
        for cls_cent in range(budget):
            min_dist_idx = np.argmin(selected_diversity_distances[cls_cent])
            if min_dist_idx not in selected_diversity_indices:
                selected_diversity_indices.append(min_dist_idx)
            else:
                sorted_idxs_min_dist = np.argsort(selected_diversity_distances[cls_cent])
                for idx in sorted_idxs_min_dist:
                    if idx not in selected_diversity_indices:
                        selected_diversity_indices.append(idx)
                        break
        print("Clustering finished")

        pbar = tqdm(total=budget, disable=args.disable_progress)
        for tot_time in range(0, len(train_dataset_all)):
            if al_sampling_type == "diversity":
                action = tot_time in selected_diversity_indices
            if action:
                cur_idx += 1
                if cur_idx >= budget:
                    break
                uncertain_index = tot_time
                lbl = train_dataset_all[int(uncertain_index)]["label"]

                diff_val = tls[lbl]
                if args.mem_decay_type == "sigmoid":
                    sigmoid_funcn = delta * (
                        1 / (1 + math.exp(-1 * alpha * diff_val + beta))
                    )
                    flip_decision, _ = checkflip(sigmoid_funcn)
                else:
                    exponential_funcn = min(math.exp(alpha*diff_val + beta), delta)
                    flip_decision, _ = checkflip(exponential_funcn)
                oracle_lbl = flip(lbl, flip_decision, num_class)

                err_lbl_score += int(oracle_lbl != lbl)
                time_last_seen_array[oracle_lbl].append(cur_idx)
                time_last_seen_array[oracle_lbl] = time_last_seen_array[oracle_lbl][
                    -mem_cnt:
                ]
                time_last_seen = get_tls_from_memory(
                    time_last_seen_array, cur_idx, mem_cnt
                )
                tls = [time_last_seen[lbl] for lbl in range(num_class)]
                selected_indices.append(uncertain_index)
                selected_indices_label.append(oracle_lbl)
                if (
                    len(selected_indices) % train_every_x == 0
                    or len(selected_indices) == budget
                ):
                    step = len(selected_indices)
                    train_dataset = train_dataset_all.select(selected_indices)
                    train_dataset = train_dataset.map(update_label, with_indices=True)
                    assert (
                        train_dataset["label"] == selected_indices_label
                    ), "Labels are not correctly updated"
                    if err_lbl_score > 0:
                        assert (
                            train_dataset["label"]
                            != train_dataset_all.select(selected_indices)["label"]
                        ), f"Error labels should be present in this case"
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=data_split_w_tokens["test"],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                    )
                    trainer.remove_callback(TensorBoardCallback)
                    trainer.remove_callback(PrinterCallback)

                    _ = trainer.train()
                    eval_score_dict = trainer.evaluate()
                    score_eval_loss.append([sd, step, eval_score_dict["eval_loss"]])
                    score_eval_f1_micro.append(
                        [sd, step, eval_score_dict["eval_f1_micro"]]
                    )
                    score_eval_f1_macro.append(
                        [sd, step, eval_score_dict["eval_f1_macro"]]
                    )
                    score_eval_f1_sadness.append(
                        [sd, step, eval_score_dict["eval_f1_sadness"]]
                    )
                    score_eval_f1_joy.append([sd, step, eval_score_dict["eval_f1_joy"]])
                    score_eval_f1_anger.append(
                        [sd, step, eval_score_dict["eval_f1_anger"]]
                    )
                    score_eval_f1_fear.append(
                        [sd, step, eval_score_dict["eval_f1_fear"]]
                    )
                    score_eval_f1_surprise.append(
                        [sd, step, eval_score_dict["eval_f1_surprise"]]
                    )
                    score_error_lbl_values.append([sd, step, selected_indices_label[:]])
                    score_train_explore.append([sd, step, tot_time])
                    score_error_lbl_score.append([sd, step, err_lbl_score])
                    score_train_text.append([sd, step, train_dataset_all.select(selected_indices)["text"]])
                    score_train_label.append([sd, step, train_dataset_all.select(selected_indices)["label"]])
                    assert len(selected_indices) == len(selected_indices_label), \
                        f"Length mismatch Indices: {len(selected_indices)}\tLabels: {len(selected_indices_label)}"
                pbar.update(1)
        pbar.close()
        print(f"Errors: {err_lbl_score}/{budget}")
        print(f"Explored: {tot_time} for selecting {budget}")
        print(f"Labels annotated: {Counter(selected_indices_label)}")
    time_taken = datetime.datetime.now() - now
    result_dict = {
        "score_eval_loss": score_eval_loss,
        "score_eval_f1_micro": score_eval_f1_micro,
        "score_eval_f1_macro": score_eval_f1_macro,
        "score_eval_f1_sadness": score_eval_f1_sadness,
        "score_eval_f1_joy": score_eval_f1_joy,
        "score_eval_f1_anger": score_eval_f1_anger,
        "score_eval_f1_fear": score_eval_f1_fear,
        "score_eval_f1_surprise": score_eval_f1_surprise,
        "score_train_explore": score_train_explore,
        "score_error_lbl_score": score_error_lbl_score,
        "score_train_text": score_train_text,
        "score_train_label": score_train_label,
        "score_error_lbl_values": score_error_lbl_values,
        "time_taken": time_taken,
        "al_model_name": model_name,
        "mem_decay": args.mem_decay,
        "seed_values": args.seed_values,
        "budget": args.budget,
        "train_every_x": args.train_every_x,
    }
    if not args.dont_save:
        file_name_prefix = args.save_prefix
        if al_sampling_type in ["diversity"]:
            file_name_prefix += f"_agg_div"
        result_store_path = os.path.join(
            "result_cikm",
            "diversity_clustering",
            args.data_type,
            f"test{file_name_prefix}_mem_dec_{alpha}_{int(beta)}.pkl",
        )
        if not os.path.isdir(os.path.dirname(result_store_path)):
            os.makedirs(os.path.dirname(result_store_path))
        with open(result_store_path, "wb") as f:
            pickle.dump(result_dict, f)
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/rpandey4/crowdrl22/data/emotions/split_train_w_ft_a_embed_no_love_upd_lbl",
        help="...",
        type=str,
    )
    parser.add_argument(
        "--use_cpu", action="store_true", help="force cpu inference in gpu cluster"
    )
    parser.add_argument(
        "--al_bert_model_name",
        default="prajjwal1/bert-mini",
        help="...",
        type=str,
    )
    parser.add_argument(
        "--al_sampling_type", default="diversity", help="Only diversity", type=str,
    )
    parser.add_argument(
        "--data_type", default="emotion", help="Test set data name. Currently supported only 'emotion'", type=str,
    )
    parser.add_argument(
        "--mem_decay",
        default="0.3,6,1",
        help="Memory decay parameters seperated by comma",
        type=str,
    )
    parser.add_argument(
        "--mem_decay_type",
        default="sigmoid",
        help="what type of decay: sigmoid or exponential",
        type=str,
    )
    parser.add_argument(
        "--tls_memory_size",
        default=3,
        help="Past K memory to remember the time last seen value",
        type=int,
    )
    parser.add_argument(
        "--seed_values",
        default="0,10,11,555,999",
        help="random seed values seperated by comma",
        type=str,
    )
    parser.add_argument(
        "--save_prefix",
        default="",
        help="any prefix to add when saving the results",
        type=str,
    )
    parser.add_argument(
        "--rl_model_save_path",
        default="",
        help="explicitly provide the save path",
        type=str,
    )
    parser.add_argument(
        "--budget", default=500, help="Budget for active learning", type=int
    )
    parser.add_argument(
        "--train_every_x", default=25, help="Update frequency to train AL", type=int
    )
    parser.add_argument(
        "--top_k_pred", default=2, help="Top k prediction to consider as possible true label", type=int
    )
    parser.add_argument(
        "--dont_save", action="store_true", help="during debug, if not saving the results"
    )
    parser.add_argument(
        "--disable_progress", action="store_true", help="during debug, if not saving the results"
    )
    args = parser.parse_args()
    print("Started Testing... Arguments:")
    print(args)
    _ = do_testing(args)
    return


if __name__ == "__main__":
    main()
