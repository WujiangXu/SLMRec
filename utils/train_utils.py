import transformers
import os
from typing import Any, Dict, List, Optional, Union
import json
import torch
import torch.nn.functional as F
import math
from transformers.trainer import *

class DistillationTrainingArguments(transformers.TrainingArguments):
    def __init__(self, *args, distill_lambda=0.001, llama_decoder_nums_student=8, llama_decoder_nums_teacher=32, distill_block=4, distill_type="other", distill_leave_layers=0, distill_type_standard="offline",
                is_cls_multiple=False,
                cls_multiple_lambda=1.0,
                kd_loss_type="cosine",
                is_cls_multiple_teacher=False,
                is_cls_multiple_student=False,
                cls_multiple_lambda_teacher=1.0,
                cls_multiple_lambda_student=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_lambda = distill_lambda
        self.llama_decoder_nums_student = llama_decoder_nums_student
        self.llama_decoder_nums_teacher = llama_decoder_nums_teacher
        self.distill_block = distill_block
        self.distill_type = distill_type
        self.distill_leave_layers = distill_leave_layers
        self.distill_type_standard = distill_type_standard
        self.is_cls_multiple=is_cls_multiple
        self.cls_multiple_lambda=cls_multiple_lambda
        self.kd_loss_type=kd_loss_type
        self.is_cls_multiple_teacher=is_cls_multiple_teacher
        self.is_cls_multiple_student=is_cls_multiple_student
        self.cls_multiple_lambda_teacher=cls_multiple_lambda_teacher
        self.cls_multiple_lambda_student=cls_multiple_lambda_student

class SLMTrainer(transformers.Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        # save output
        # 以追加模式打开文件，并将字符串追加到文件
        with open(os.path.join(self.args.output_dir,"log.txt"), 'a') as file:
            # print("logger output:{}".format(output))
            json.dump(output, file)
            file.write('\n')  # 添加换行符

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

class DistillationTrainer(transformers.Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        if teacher_model is not None:
            self._move_model_to_device(self.teacher, self.model.llama_model.device)
            self.teacher.eval()
        # self.logger = logger

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


class RecDistillationTrainer(DistillationTrainer,SLMTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.distill_type_standard == "offline":
            # compute student output
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs_student = model(**inputs)
            try:
                if(torch.max(outputs_student['data_type']).item() ==0):
                    student_loss = outputs_student['loss']
                    # compute teacher output
                    with torch.no_grad():
                        outputs_teacher = self.teacher(**inputs)

                    # assert size
                    # assert outputs_student.logits.size() == outputs_teacher.logits.size()
                    loss_distill = 0
                    teacher_output_states = outputs_teacher['teacher_output_states']
                    student_hidden_states = outputs_student['student_output_states']
                    if student_hidden_states is not None:
                        if self.args.distill_type=="align": # block-wise alignment
                            for i in range(1,self.args.distill_block+1): # 1-4 block
                                cosine_sim = F.cosine_similarity(student_hidden_states[(self.args.llama_decoder_nums_student//self.args.distill_block)*i][:,-1], teacher_output_states[(self.args.llama_decoder_nums_teacher//self.args.distill_block)*i][:,-1], dim=1)
                                # 将相似度转换为损失
                                l2_distance = torch.norm(student_hidden_states[(self.args.llama_decoder_nums_student//self.args.distill_block)*i][:,-1] - teacher_output_states[(self.args.llama_decoder_nums_teacher//self.args.distill_block)*i][:,-1], dim=1, p=2).mean()
                                # print("l2_distance:{}".format(l2_distance))
                                loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda + l2_distance * 0.1
                        # elif self.args.distill_type=="self":
                        #     for i in range(1,self.args.llama_decoder_nums_student//2+1): # 1-4 block
                        #         cosine_sim = F.cosine_similarity(student_hidden_states[i][:,-1], teacher_output_states[2*i][:,-1], dim=1)
                        #         # 将相似度转换为损失
                        #         loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda
                        # elif self.args.distill_type=="remove_last": # interval some layers to align except last hidden states
                        #     # 待实现间隔多少层，目前是直接舍弃最后一层
                        #     for i in range(1,self.args.llama_decoder_nums_student):
                        #         map_nums = math.ceil(self.args.llama_decoder_nums_student / self.args.llama_decoder_nums_teacher)
                        #         cosine_sim = F.cosine_similarity(student_hidden_states[i][:,-1], teacher_output_states[self.args.llama_decoder_nums_teacher-map_nums*(self.args.llama_decoder_nums_student-i-1)][:,-1], dim=1)
                        #         # 将相似度转换为损失
                        #         loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda
                        print("loss_distill:{}".format(loss_distill))
                        # self.logger.info(f"  loss_distill: {loss_distill}")
                        loss_distill_dict = {"loss_distill":loss_distill.item()}
                        self.log(loss_distill_dict)
                    else:
                        loss_distill = 0
                    if self.args.is_cls_multiple:
                        if outputs_student['loss_cls_multiple'] is not None:
                            loss_multiple = outputs_student['loss_cls_multiple'] * self.args.cls_multiple_lambda
                            student_loss = student_loss + loss_multiple
                            loss_multiple_dict = {"loss_multiple":loss_multiple.item()}
                            self.log(loss_multiple_dict)
                    loss = student_loss + loss_distill
            except:
                loss = outputs_student['loss']
        elif self.args.distill_type_standard=="online":
            # compute student output
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs_student = model(**inputs)
            try:
                if torch.max(outputs_student['data_type']).item() ==0:
                    student_loss = outputs_student['loss']
                    teacher_output_states = outputs_student['teacher_output_states']
                    student_hidden_states = outputs_student['student_output_states']

                    # assert size
                    # assert outputs_student.logits.size() == outputs_teacher.logits.size()
                    loss_distill = 0
                    if self.args.kd_loss_type == "cosine":
                        if teacher_output_states is not None and student_hidden_states is not None:
                            if self.args.distill_type=="align": # block-wise alignment
                                for i in range(1,self.args.distill_block+1): # 1-4 block
                                    cosine_sim = F.cosine_similarity(student_hidden_states[(self.args.llama_decoder_nums_student//self.args.distill_block)*i][:,-1], teacher_output_states[(self.args.llama_decoder_nums_teacher//self.args.distill_block)*i][:,-1], dim=1)
                                    # 将相似度转换为损失
                                    l2_distance = torch.norm(student_hidden_states[(self.args.llama_decoder_nums_student//self.args.distill_block)*i][:,-1] - teacher_output_states[(self.args.llama_decoder_nums_teacher//self.args.distill_block)*i][:,-1], dim=1, p=2).mean()
                                    # print("l2_distance:{}".format(l2_distance))
                                    loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda + l2_distance * 0.1
                                    # loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda
                            # elif self.args.distill_type=="self":
                            #     for i in range(1,self.args.llama_decoder_nums_student//2+1): # 1-4 block
                            #         cosine_sim = F.cosine_similarity(student_hidden_states[i][:,-1], teacher_output_states[2*i][:,-1], dim=1)
                            #         # 将相似度转换为损失
                            #         loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda
                            # elif self.args.distill_type=="remove_last": # interval some layers to align except last hidden states
                            #     # 待实现间隔多少层，目前是直接舍弃最后一层
                            #     for i in range(1,self.args.llama_decoder_nums_student):
                            #         map_nums = math.ceil(self.args.llama_decoder_nums_student / self.args.llama_decoder_nums_teacher)
                            #         cosine_sim = F.cosine_similarity(student_hidden_states[i][:,-1], teacher_output_states[self.args.llama_decoder_nums_teacher-map_nums*(self.args.llama_decoder_nums_student-i-1)][:,-1], dim=1)
                            #         # 将相似度转换为损失
                            #         loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda
                    
                            # self.logger.info(f"  loss_distill: {loss_distill}")
                            loss_distill_dict = {"loss_distill":loss_distill.item()}
                            self.log(loss_distill_dict)
                        else:
                            loss_distill = 0
                    elif self.args.kd_loss_type == "logit":
                        for i in range(0,self.args.distill_block):
                            if outputs_student['logits_teacher'] is not None and outputs_student['logits_student'] is not None: 
                                logits_teacher_tmp, logits_student_tmp = outputs_student['logits_teacher'][i], outputs_student['logits_student'][i] 
                                # 教师logits通过softmax获取概率分布，并取对数
                                teacher_probs = F.softmax(logits_teacher_tmp, dim=1)
                                # teacher_probs_with_log = torch.log(teacher_probs + 1e-9)  # 防止log(0)，可以加一个小的epsilon

                                # 学生logits直接传入KLDivLoss，因为PyTorch中的KLDivLoss预计第一个参数是对数形式
                                student_probs_with_log = F.log_softmax(logits_student_tmp, dim=1)

                                # 定义KL散度损失, reduction='batchmean'意味着要在批次中取平均
                                kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

                                # 计算损失值
                                loss_distill = loss_distill + kl_loss(student_probs_with_log, teacher_probs.detach()) * self.args.distill_lambda
                                loss_distill_dict = {"loss_distill_kl":loss_distill.item()}
                                self.log(loss_distill_dict)

                    if self.args.is_cls_multiple_teacher:
                        if outputs_student['loss_cls_multiple_teacher'] is not None:
                            loss_multiple_teacher = outputs_student['loss_cls_multiple_teacher'] * self.args.cls_multiple_lambda_teacher
                            student_loss = student_loss + loss_multiple_teacher
                            loss_multiple_teacher_dict = {"loss_multiple_teacher":loss_multiple_teacher.item()}
                            self.log(loss_multiple_teacher_dict)

                    if self.args.is_cls_multiple_student:
                        if outputs_student['loss_cls_multiple_student'] is not None:
                            loss_multiple_student = outputs_student['loss_cls_multiple_student'] * self.args.cls_multiple_lambda_student
                            student_loss = student_loss + loss_multiple_student
                            loss_multiple_student_dict = {"loss_multiple_student":loss_multiple_student.item()}
                            self.log(loss_multiple_student_dict)
                    
                    loss = student_loss + loss_distill
            except:
                loss = outputs_student['loss']
        return (loss, outputs_student) if return_outputs else loss