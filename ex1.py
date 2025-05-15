# ex1.py — MRPC Fine-Tuning using argparse
import wandb
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EvalPrediction, set_seed
)
from sklearn.metrics import accuracy_score

# ------------------ Metric Function ------------------
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# ------------------ Preprocessing Function ------------------
def preprocess(example, do_padding=True):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        padding='max_length' if do_padding else False,
        max_length=tokenizer.model_max_length
    )



# ------------------ Main Training & Prediction Logic ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"

    training_args = TrainingArguments(
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_steps=1,
        report_to=["wandb"]
    )

    set_seed(42)
    run = wandb.init(project="mrpc_ex1", config=vars(args))

    dataset = load_dataset("nyu-mll/glue", "mrpc")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset["train"] = dataset["train"].map(lambda x: preprocess(x, do_padding=True), batched=True)
    dataset["validation"] = dataset["validation"].map(lambda x: preprocess(x, do_padding=True), batched=True)
    dataset["test"] = dataset["test"].map(lambda x: preprocess(x, do_padding=False), batched=True)

    if args.max_train_samples != -1:
        max_train = min(args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(max_train))
    if args.max_eval_samples != -1:
        max_eval = min(args.max_eval_samples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(max_eval))
    if args.max_predict_samples != -1:
        max_predict = min(args.max_predict_samples, len(dataset["test"]))
        dataset["test"] = dataset["test"].select(range(max_predict))

    config = AutoConfig.from_pretrained(args.model_path, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)
    use_collator = args.do_train or not args.do_predict
    data_collator = DataCollatorWithPadding(tokenizer) if use_collator else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if args.do_train else None,
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model(f"checkpoint_lr{args.lr}_bs{args.batch_size}_ep{args.num_train_epochs}")
        metrics = trainer.evaluate()
        val_acc = metrics.get("eval_accuracy", -1)

        with open("res.txt", "a") as f:
            f.write(
                f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {val_acc:.4f}\n")
        wandb.log({"val_accuracy": val_acc})

    if args.do_predict:
        model.eval()
        predictions = trainer.predict(dataset["test"])
        preds = np.argmax(predictions.predictions, axis=1)

        from sklearn.metrics import accuracy_score
        labels = dataset["test"]["label"]
        test_acc = accuracy_score(labels, preds)

        raw_test = load_dataset("nyu-mll/glue", "mrpc")["test"]

        with open("predictions.txt", "w", encoding="utf-8") as f:
            for i in range(len(preds)):
                s1 = raw_test[i]["sentence1"]
                s2 = raw_test[i]["sentence2"]
                pred = preds[i]
                f.write(f"{s1}###{s2}###{pred}\n")
        #SECTION FOR COMPARING THE OUTPUTS OF THE BEST VS THE WORST CONFIGURATIONS COMMENTED OUT
        # with open("res.txt", "a") as f:
        #     f.write(
        #         f"TEST — lr={args.lr}, bs={args.batch_size}, epochs={args.num_train_epochs} => test_acc={test_acc:.4f}\n")

        # Access original (non-tokenized) test set for sentences and labels
        # raw_test = load_dataset("nyu-mll/glue", "mrpc")["test"]
        # true_labels = dataset["test"]["label"]

        # # Write detailed comparison
        # with open("compare.txt", "w", encoding="utf-8") as f:
        #     for i in range(len(preds)):
        #         s1 = raw_test[i]["sentence1"]
        #         s2 = raw_test[i]["sentence2"]
        #         label = true_labels[i]
        #         pred = preds[i]
        # 
        #         f.write(f"Example {i}\n")
        #         f.write(f"Sentence 1: {s1}\n")
        #         f.write(f"Sentence 2: {s2}\n")
        #         f.write(f"True Label:     {label}\n")
        #         f.write(f"Predicted Label:{pred}\n")
        #         f.write("-" * 40 + "\n")

        wandb.log({"test_accuracy": test_acc})

    wandb.finish()

if __name__ == "__main__":
    main()
