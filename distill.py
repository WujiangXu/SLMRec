import os
import sys
from typing import List
import argparse

import fire
import torch
import pickle
import numpy as np
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from model import LLM4Rec,LLM4RecText,LLM4RecDistill,LLM4RecTeacher,LLM4RecStudent
from utils.data_utils import *
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, AUC, getLabel, compute_metrics
from utils.train_utils import CDRTrainer,RecDistillationTrainer,DistillationTrainingArguments

def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "",
    train_stargy: str = "lora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    interval_nums: int = 0,
    drop_type: str="trune",
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100, 
    max_steps: int = -1,
    save_steps: int = 100,
    eval_steps: int = 100,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    teacher_resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    student_resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",
    domain_type: str = "cloths",
    llama_decoder_nums_teacher: int = 32,
    llama_decoder_nums_student: int = 8,
    distill_block: int = 4,
    distill_lambda: float = 1.0,
    distill_type: str = "other",
    distill_type_standard: str = "offline",
    is_cls_multiple: bool = False, 
    train_eval_type: str = "train",
    cls_multiple_lambda: float = 1.0,
    kd_loss_type: str = "cosine",
    is_cls_multiple_teacher: bool = False,
    is_cls_multiple_student: bool = False,
    cls_multiple_lambda_teacher: float = 1.0,
    cls_multiple_lambda_student: float = 1.0,

):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"cache_dir: {cache_dir}\n"
            f"output_dir: {output_dir}\n"
            f"task_type: {task_type}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"teacher_resume_from_checkpoint: {teacher_resume_from_checkpoint}\n"
            f"student_resume_from_checkpoint: {student_resume_from_checkpoint}\n"
            f"llama_decoder_nums_teacher: {llama_decoder_nums_teacher}\n"
            f"llama_decoder_nums_student: {llama_decoder_nums_student}\n"
            f"distill_lambda: {distill_lambda}\n"
            f"distill_block: {distill_block}\n"
            f"domain_type: {domain_type}\n"
            f"distill_type: {distill_type}\n"
            f"distill_type_standard: {distill_type_standard}\n"
            f"is_cls_multiple: {is_cls_multiple}\n"
            f"cls_multiple_lambda: {cls_multiple_lambda}\n"
            f"kd_loss_type: {kd_loss_type}\n"
            f"is_cls_multiple_teacher: {is_cls_multiple_teacher}\n"
            f"is_cls_multiple_student:{is_cls_multiple_student}\n"
            f"cls_multiple_lambda_teacher:{cls_multiple_lambda_teacher}\n"
            f"cls_multiple_lambda_student:{cls_multiple_lambda_student}\n"
        )
    # assert (
    #     base_model
    # ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # choose from cloths and movies
    item_embed = pickle.load(open('./sasrec_{}/sasrec_item.pkl'.format(domain_type), 'rb'))['item_embedding']
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    #args.include_inputs_for_metrics --> true
    datasetTrain = LLMDataset(item_size=999, max_seq_length=30,data_type='train',csv_path="./dataset/sequential/{}.csv".format(domain_type))
    datasetVal = LLMDataset(item_size=999, max_seq_length=30,data_type='valid',csv_path="./dataset/sequential/{}.csv".format(domain_type))
    datasetTest = LLMDataset(item_size=999, max_seq_length=30,data_type='test',csv_path="./dataset/sequential/{}.csv".format(domain_type))
    data_collator = SequentialCollator()
    if save_steps<0:
        save_strategy = "epoch"
    else:
        save_strategy = "steps"
    if eval_steps<0:
        evaluation_strategy = "epoch"
    else:
        evaluation_strategy = "steps"

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    # print(eval_dataset)
    # trainer.evaluate(eval_dataset=eval_dataset)
    if distill_type_standard=="offline":
        model_teacher =  LLM4RecTeacher(
            base_model=base_model,
            task_type=task_type,
            cache_dir=cache_dir,
            input_dim=128,
            output_dim=0,
            interval_nums=interval_nums,
            drop_type=drop_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=device_map,
            instruction_text=prompter.generate_prompt(task_type),
            train_stargy = train_stargy,
            user_embeds=None,
            input_embeds=item_embed,
            seq_len=30,
            llama_decoder_nums=llama_decoder_nums_teacher,
            # llama_decoder_nums_teacher=llama_decoder_nums_teacher,
            # llama_decoder_nums_student=llama_decoder_nums_student,
            # distill_lambda=distill_lambda,
        )      
        model = LLM4RecStudent(
            base_model=base_model,
            task_type=task_type,
            cache_dir=cache_dir,
            input_dim=128,
            output_dim=0,
            interval_nums=interval_nums,
            drop_type=drop_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=device_map,
            instruction_text=prompter.generate_prompt(task_type),
            train_stargy = train_stargy,
            user_embeds=None,
            input_embeds=item_embed,
            seq_len=30,
            llama_decoder_nums=llama_decoder_nums_student,
            distill_block=distill_block,
            is_cls_multiple=is_cls_multiple,
            # llama_decoder_nums_teacher=llama_decoder_nums_teacher,
            # llama_decoder_nums_student=llama_decoder_nums_student,
            # distill_lambda=distill_lambda,
        )
        trainer = RecDistillationTrainer(#transformers.Trainer(
            teacher_model=model_teacher,
            model=model,
            train_dataset=datasetTrain,
            eval_dataset=datasetVal,
            args=DistillationTrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                include_inputs_for_metrics = True,
                gradient_accumulation_steps=1, # change it
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                dataloader_num_workers=20,
                per_device_eval_batch_size = 1024,
                remove_unused_columns = False,
                fp16=True,
                logging_steps=1,
                optim="adamw_torch",
                metric_for_best_model="mrr",
                # evaluation_strategy="steps", #if val_set_size > 0 else "no",
                evaluation_strategy=evaluation_strategy, # epoch
                save_strategy=save_strategy,
                max_steps=max_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                lr_scheduler_type="cosine",
                logging_dir = output_dir,
                output_dir=output_dir,
                save_total_limit=2,
                load_best_model_at_end=False,#True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                # use_reentrant=True,
                group_by_length=group_by_length,
                report_to="tensorboard",
                # hub_strategy="checkpoint",
                run_name=None,
                distill_lambda=distill_lambda,
                llama_decoder_nums_student=llama_decoder_nums_student,
                llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                distill_block=distill_block,
                distill_type=distill_type,
                distill_type_standard=distill_type_standard,
                is_cls_multiple=is_cls_multiple,
                cls_multiple_lambda=cls_multiple_lambda,
                kd_loss_type=kd_loss_type,
                # ddp_find_unused_parameters=True,
            ),
            data_collator=data_collator,
            compute_metrics = compute_metrics,
        )
        trainer._load_from_checkpoint(teacher_resume_from_checkpoint,model=model_teacher)
        model_teacher.eval()
    elif distill_type_standard=="online":
        model = LLM4RecDistill(
            base_model=base_model,
            task_type=task_type,
            cache_dir=cache_dir,
            input_dim=128,
            output_dim=0,
            interval_nums=interval_nums,
            drop_type=drop_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=device_map,
            instruction_text=prompter.generate_prompt(task_type),
            train_stargy = train_stargy,
            user_embeds=None,
            item_embed=item_embed,
            seq_len=30,
            llama_decoder_nums=llama_decoder_nums_teacher,
            llama_decoder_nums_teacher=llama_decoder_nums_teacher,
            llama_decoder_nums_student=llama_decoder_nums_student,
            distill_lambda=distill_lambda,
            distill_block=distill_block,
            is_cls_multiple_teacher=is_cls_multiple_teacher,
            is_cls_multiple_student=is_cls_multiple_student,
        ) 
        trainer = RecDistillationTrainer(
            teacher_model=None, #cover in student model
            model=model,
            train_dataset=datasetTrain,
            eval_dataset=datasetVal,
            args=DistillationTrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                include_inputs_for_metrics = True,
                gradient_accumulation_steps=1, # change it
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                dataloader_num_workers=20,
                per_device_eval_batch_size = 1024,
                remove_unused_columns = False,
                fp16=True,
                logging_steps=1,
                optim="adamw_torch",
                metric_for_best_model="mrr",
                # evaluation_strategy="steps", #if val_set_size > 0 else "no",
                evaluation_strategy=evaluation_strategy, # epoch
                save_strategy=save_strategy,
                max_steps=max_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                lr_scheduler_type="cosine",
                logging_dir = output_dir,
                output_dir=output_dir,
                save_total_limit=2,
                load_best_model_at_end=False,#True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                # use_reentrant=True,
                group_by_length=group_by_length,
                report_to="tensorboard",
                # hub_strategy="checkpoint",
                run_name=None,
                distill_lambda=distill_lambda,
                llama_decoder_nums_student=llama_decoder_nums_student,
                llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                distill_block=distill_block,
                distill_type=distill_type,
                distill_type_standard=distill_type_standard,
                is_cls_multiple=is_cls_multiple,
                is_cls_multiple_teacher=is_cls_multiple_teacher,
                is_cls_multiple_student=is_cls_multiple_student,
                cls_multiple_lambda=cls_multiple_lambda,
                kd_loss_type=kd_loss_type,
                cls_multiple_lambda_teacher=cls_multiple_lambda_teacher,
                cls_multiple_lambda_student=cls_multiple_lambda_student,
                # ddp_find_unused_parameters=True,
            ),
            data_collator=data_collator,
            compute_metrics = compute_metrics,
        )
    # if distill_type=="self":
    #     trainer._load_from_checkpoint(teacher_resume_from_checkpoint)
    # def print_attributes(obj):
    #     for attribute_name in dir(obj):
    #         if not attribute_name.startswith('__'):
    #             attribute_value = getattr(obj, attribute_name)
    #             print(f"{attribute_name}: {attribute_value}")

    # 打印model的所有属性
    # print("model的属性:")
    # print_attributes(trainer.model.llama_model)
    # print(hasattr(trainer.model.llama_model, "active_adapter"))
    # print(hasattr(trainer.model.llama_model, "load_adapter"))
    # if hasattr(model, "base_model") and getattr(model.base_model, "is_8bit_serializable", False):
    # print()model_student 
    # print("llama_model model:{}".format(hasattr(trainer.model, "llama_model")))    
    # print("base_model :{}".format(hasattr(trainer.model.llama_model, "base_model")))    
    # print("8bit :{}".format(getattr(trainer.model.llama_model.base_model, "is_8bit_serializable", False)))  
    # print(trainer.state.best_model_checkpoint)  
    if train_eval_type=="train":
        # trainer._load_from_checkpoint(student_resume_from_checkpoint)
        trainer.train()
        # trainer.evaluate(eval_dataset=datasetVal)
        best_checkpoint_path = trainer.state.best_model_checkpoint
        # trainer._load_from_checkpoint(trainer.state.best_model_checkpoint)
    elif train_eval_type=="eval":
        trainer._load_from_checkpoint(student_resume_from_checkpoint)
    
    if train_eval_type=="train":
        # reload the model and path
        if distill_type_standard=="offline":
            model_teacher =  LLM4RecTeacher(
                base_model=base_model,
                task_type=task_type,
                cache_dir=cache_dir,
                input_dim=128,
                output_dim=0,
                interval_nums=interval_nums,
                drop_type=drop_type,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                device_map=device_map,
                instruction_text=prompter.generate_prompt(task_type),
                train_stargy = train_stargy,
                user_embeds=None,
                input_embeds=item_embed,
                seq_len=30,
                llama_decoder_nums=llama_decoder_nums_teacher,
                # llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                # llama_decoder_nums_student=llama_decoder_nums_student,
                # distill_lambda=distill_lambda,
            )      
            model = LLM4RecStudent(
                base_model=base_model,
                task_type=task_type,
                cache_dir=cache_dir,
                input_dim=128,
                output_dim=0,
                interval_nums=interval_nums,
                drop_type=drop_type,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                device_map=device_map,
                instruction_text=prompter.generate_prompt(task_type),
                train_stargy = train_stargy,
                user_embeds=None,
                input_embeds=item_embed,
                seq_len=30,
                llama_decoder_nums=llama_decoder_nums_student,
                distill_block=distill_block,
                is_cls_multiple=is_cls_multiple,
                # llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                # llama_decoder_nums_student=llama_decoder_nums_student,
                # distill_lambda=distill_lambda,
            )
            trainer = RecDistillationTrainer(#transformers.Trainer(
                teacher_model=model_teacher,
                model=model,
                train_dataset=datasetTrain,
                eval_dataset=datasetVal,
                args=DistillationTrainingArguments(
                    per_device_train_batch_size=micro_batch_size,
                    include_inputs_for_metrics = True,
                    gradient_accumulation_steps=1, # change it
                    warmup_steps=warmup_steps,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    dataloader_num_workers=20,
                    per_device_eval_batch_size = 1024,
                    remove_unused_columns = False,
                    fp16=True,
                    logging_steps=1,
                    optim="adamw_torch",
                    metric_for_best_model="mrr",
                    # evaluation_strategy="steps", #if val_set_size > 0 else "no",
                    evaluation_strategy=evaluation_strategy, # epoch
                    save_strategy=save_strategy,
                    eval_steps=eval_steps,
                    save_steps=save_steps,
                    max_steps=max_steps,
                    lr_scheduler_type="cosine",
                    logging_dir = output_dir,
                    output_dir=output_dir,
                    save_total_limit=2,
                    load_best_model_at_end=False,#True if val_set_size > 0 else False,
                    ddp_find_unused_parameters=False if ddp else None,
                    # use_reentrant=True,
                    group_by_length=group_by_length,
                    report_to="tensorboard",
                    # hub_strategy="checkpoint",
                    run_name=None,
                    distill_lambda=distill_lambda,
                    llama_decoder_nums_student=llama_decoder_nums_student,
                    llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                    distill_block=distill_block,
                    distill_type=distill_type,
                    distill_type_standard=distill_type_standard,
                    is_cls_multiple=is_cls_multiple,
                    cls_multiple_lambda=cls_multiple_lambda,
                    kd_loss_type=kd_loss_type,
                    # ddp_find_unused_parameters=True,
                ),
                data_collator=data_collator,
                compute_metrics = compute_metrics,
            )
            trainer._load_from_checkpoint(teacher_resume_from_checkpoint,model=model_teacher)
            model_teacher.eval()
        elif distill_type_standard=="online":
            model = LLM4RecDistill(
                base_model=base_model,
                task_type=task_type,
                cache_dir=cache_dir,
                input_dim=128,
                output_dim=0,
                interval_nums=interval_nums,
                drop_type=drop_type,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                device_map=device_map,
                instruction_text=prompter.generate_prompt(task_type),
                train_stargy = train_stargy,
                user_embeds=None,
                item_embed=item_embed,
                seq_len=30,
                llama_decoder_nums=llama_decoder_nums_teacher,
                llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                llama_decoder_nums_student=llama_decoder_nums_student,
                distill_lambda=distill_lambda,
                distill_block=distill_block,
                is_cls_multiple_teacher=is_cls_multiple_teacher,
                is_cls_multiple_student=is_cls_multiple_student,
            ) 
            trainer = RecDistillationTrainer(
                teacher_model=None, #cover in student model
                model=model,
                train_dataset=datasetTrain,
                eval_dataset=datasetVal,
                args=DistillationTrainingArguments(
                    per_device_train_batch_size=micro_batch_size,
                    include_inputs_for_metrics = True,
                    gradient_accumulation_steps=1, # change it
                    warmup_steps=warmup_steps,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    dataloader_num_workers=20,
                    per_device_eval_batch_size = 1024,
                    remove_unused_columns = False,
                    fp16=True,
                    logging_steps=1,
                    optim="adamw_torch",
                    metric_for_best_model="mrr",
                    # evaluation_strategy="steps", #if val_set_size > 0 else "no",
                    evaluation_strategy=evaluation_strategy, # epoch
                    save_strategy=save_strategy,
                    eval_steps=eval_steps,
                    save_steps=save_steps,
                    max_steps=max_steps,
                    lr_scheduler_type="cosine",
                    logging_dir = output_dir,
                    output_dir=output_dir,
                    save_total_limit=2,
                    load_best_model_at_end=False,#True if val_set_size > 0 else False,
                    ddp_find_unused_parameters=False if ddp else None,
                    # use_reentrant=True,
                    group_by_length=group_by_length,
                    report_to="tensorboard",
                    # hub_strategy="checkpoint",
                    run_name=None,
                    distill_lambda=distill_lambda,
                    llama_decoder_nums_student=llama_decoder_nums_student,
                    llama_decoder_nums_teacher=llama_decoder_nums_teacher,
                    distill_block=distill_block,
                    distill_type=distill_type,
                    distill_type_standard=distill_type_standard,
                    is_cls_multiple=is_cls_multiple,
                    is_cls_multiple_teacher=is_cls_multiple_teacher,
                    is_cls_multiple_student=is_cls_multiple_student,
                    cls_multiple_lambda=cls_multiple_lambda,
                    kd_loss_type=kd_loss_type,
                    cls_multiple_lambda_teacher=cls_multiple_lambda_teacher,
                    cls_multiple_lambda_student=cls_multiple_lambda_student,
                    # ddp_find_unused_parameters=True,
                ),
                data_collator=data_collator,
                compute_metrics = compute_metrics,
            )
        trainer._load_from_checkpoint(best_checkpoint_path)    
    pred_out = trainer.predict(test_dataset=datasetTest)
    # 可以这样访问 metrics 字典
    output_data = {}
    if pred_out.metrics is not None:
        for metric_name, metric_value in pred_out.metrics.items():
            print(f"{metric_name}: {metric_value}")
            output_data[metric_name] = metric_value

    # Write the output data to a file
    with open(os.path.join(output_dir,"log.txt"), 'a') as file:
        json.dump(output_data, file)
    # train postprocess load best result
    # clean

    # model.llama_model.save_pretrained(output_dir)
    # model_path = os.path.join(output_dir, "adapter.pth")
    # if task_type == 'general':
    #     user_proj, input_proj, score = model.user_proj.state_dict(), model.input_proj.state_dict(), model.score.state_dict()
    #     torch.save({'user_proj': user_proj, 'input_proj': input_proj, 'score': score}, model_path)
    # elif task_type == 'sequential':
    #     input_proj, score = model.input_proj.state_dict(), model.score.state_dict()
    #     torch.save({'input_proj': input_proj, 'score': score}, model_path)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
