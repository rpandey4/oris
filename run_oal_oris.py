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
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, PrinterCallback, Trainer,
                          TrainingArguments)
from transformers.integrations import TensorBoardCallback


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
        tls_avg = sum(tls_mem_relative) / non_zero_tls_mem
        time_last_seen[lbl] = tls_avg  # - 1.5
    return time_last_seen


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def unc_schedule(step, initial_lr=0.9, decay_rate=5, clip_min=0.1, num_steps=10):
    """
    Learning rate schedule that decays exponentially from an initial learning rate
    to a final learning rate over a fixed number of steps.

    Parameters:
        - step: the current training step (integer)
        - initial_lr: the initial learning rate (float, default=0.9)
        - decay_rate: the rate at which to decay the learning rate (float, default=0.1)
        - num_steps: the total number of steps in the schedule (integer, default=10)

    Returns:
        - lr: the current learning rate (float)
    """
    decay_steps = num_steps / np.log(1.0 / decay_rate)
    lr = initial_lr * decay_rate ** (step / decay_steps)
    lr = max(lr, clip_min)  # clip to a minimum value
    return lr


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

    def select_action(state):
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(state.shape[0], 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_cpu:
        print("Forcing to CPU only inference")
        device = torch.device("cpu")
    print(device)

    # Loading RL model
    if args.rl_model_save_path == "":
        rl_model_save_path = f"model/delta_8/episode_err_{args.rl_save_ckpt:05d}.pt"
    else:
        rl_model_save_path = args.rl_model_save_path
    if torch.cuda.is_available():
        model_weights_dict = torch.load(rl_model_save_path)
    else:
        model_weights_dict = torch.load(
            rl_model_save_path, map_location=torch.device("cpu")
        )

    n_actions = 2
    n_observations = 305

    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(model_weights_dict["model_state_dict"])
    policy_net.eval()
    # Finished loading RL model

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
    decay_params = [float(x) for x in args.mem_decay.split(",")]
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
        pbar = tqdm(total=budget, disable=args.disable_progress)
        assert al_sampling_type in [
            "unc",
            "rl",
            "unc_rl",
            "random",
        ], f"Incorrect sampling strategy Choose among unc, rl, random, or unc_rl"
        assert args.mem_decay_type in [
            "sigmoid",
            "exponential",
        ], f"Only sigmoid and exponential decays are available"
        unc_th_list = []
        unc_step = 0
        if al_sampling_type in ["unc", "unc_rl"]:
            num_unc_steps = (budget // train_every_x) + 1
            if not args.unc_decay:
                unc_th_list = [args.unc_th] * num_unc_steps
            else:
                unc_th_list = [
                    unc_schedule(
                        step=s, initial_lr=args.unc_th, num_steps=num_unc_steps
                    )
                    for s in range(num_unc_steps)
                ]

        for tot_time in range(0, len(train_dataset_all)):
            if al_sampling_type == "unc":
                pred = trainer.predict(train_dataset_all.select([tot_time]))
                unc_score = entropy(
                    softmax(pred.predictions, axis=1), axis=1, base=num_class
                )[0]
                action = unc_score >= unc_th_list[unc_step]
            elif al_sampling_type == "rl":
                input_feat = train_dataset_all[tot_time]["feat"]
                observation = np.concatenate((input_feat, tls))
                state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = select_action(state).item() == 1
            elif al_sampling_type == "random":
                action = random.random() < 0.5  # random.choice((True, False))
            else:
                raise ValueError("al_sampling_type should be either unc, rl, or random")
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
                    exponential_funcn = min(math.exp(alpha * diff_val + beta), delta)
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
                    if al_sampling_type == "unc" and args.unc_decay:
                        unc_step += 1
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
                        train_dataset=train_dataset,  # tokenized_data_split["train"],
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
                    score_train_text.append(
                        [sd, step, train_dataset_all.select(selected_indices)["text"]]
                    )
                    score_train_label.append(
                        [sd, step, train_dataset_all.select(selected_indices)["label"]]
                    )
                    assert len(selected_indices) == len(
                        selected_indices_label
                    ), f"Length mismatch Indices: {len(selected_indices)}\tLabels: {len(selected_indices_label)}"
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
        "rl_save_ckpt": args.rl_save_ckpt,
        "mem_decay": args.mem_decay,
        "seed_values": args.seed_values,
        "budget": args.budget,
        "train_every_x": args.train_every_x,
    }
    if not args.dont_save:
        file_name_prefix = args.save_prefix
        if al_sampling_type == "unc":
            file_name_prefix += f"_unc_{args.unc_th:.1f}"
        if al_sampling_type == "rl":
            file_name_prefix += "_rl"
        if al_sampling_type == "random":
            file_name_prefix += "_random"
        result_store_path = os.path.join(
            "result_cikm",
            args.data_type,
            f"test{file_name_prefix}_mem_dec_{alpha}_{int(beta)}_{now.strftime('%Y_%m_%d')}.pkl",
        )
        if not os.path.isdir(os.path.dirname(result_store_path)):
            os.makedirs(os.path.dirname(result_store_path))
        with open(result_store_path, "wb") as f:
            pickle.dump(result_dict, f)
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data/emotion_twitter", help="data path", type=str,
    )
    parser.add_argument(
        "--use_cpu", action="store_true", help="force cpu inference in gpu cluster"
    )
    parser.add_argument(
        "--unc_decay", action="store_true", help="whether to schedule uncertainty decay"
    )
    parser.add_argument(
        "--validation_warmup",
        action="store_true",
        help="whether to use validation data to warmup the al model",
    )
    parser.add_argument(
        "--al_bert_model_name", default="prajjwal1/bert-mini", help="...", type=str,
    )
    parser.add_argument(
        "--al_sampling_type", default="rl", help="One of unc, rl, or random", type=str,
    )
    parser.add_argument(
        "--data_type",
        default="emotion",
        help="Test set data name. Currently supported only 'emotion'",
        type=str,
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
        "--rl_save_ckpt", default=9394, help="RL Saved Model Checkpoint", type=int
    )
    parser.add_argument(
        "--budget", default=500, help="Budget for active learning", type=int
    )
    parser.add_argument(
        "--unc_th",
        default=0.9,
        help="Threshold for stream-based uncertainty sampling",
        type=float,
    )
    parser.add_argument(
        "--train_every_x", default=25, help="Update frequency to train AL", type=int
    )
    parser.add_argument(
        "--dont_save",
        action="store_true",
        help="during debug, if not saving the results",
    )
    parser.add_argument(
        "--disable_progress",
        action="store_true",
        help="during debug, if not saving the results",
    )
    args = parser.parse_args()
    print("Started Online Active Learning... Arguments:")
    print(args)
    _ = do_testing(args)
    return


if __name__ == "__main__":
    main()
