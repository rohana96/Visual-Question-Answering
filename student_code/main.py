"""
------------
Run baseline:
------------

python student_code/main.py --model simple --train_image_dir data/train2014 \
--train_question_path data/MultipleChoice_mscoco_train2014_questions.json \
--train_annotation_path data/mscoco_train2014_annotations.json \
--test_image_dir data/val2014 \
--test_question_path data/MultipleChoice_mscoco_val2014_questions.json \
--test_annotation_path data/mscoco_val2014_annotations.json \
--log_validation
--exp_name simple

----------------
Run coattention:
----------------

python student_code/main.py --model coattention --train_image_dir data/train2014 \
--train_question_path data/MultipleChoice_mscoco_train2014_questions.json \
--train_annotation_path data/mscoco_train2014_annotations.json \
--test_image_dir data/val2014 \
--test_question_path data/MultipleChoice_mscoco_val2014_questions.json \
--test_annotation_path data/mscoco_val2014_annotations.json \
--log_validation
--exp_name coattention


----------------
Run cosine coattention:
----------------

python student_code/main.py --model coattention --train_image_dir data/train2014 \
--train_question_path data/MultipleChoice_mscoco_train2014_questions.json \
--train_annotation_path data/mscoco_train2014_annotations.json \
--test_image_dir data/val2014 \
--test_question_path data/MultipleChoice_mscoco_val2014_questions.json \
--test_annotation_path data/mscoco_val2014_annotations.json \
--log_validation \
--exp_name cosine_coattention

"""

import argparse
import sys

sys.path.append('../../VQA')
from simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from coattention_experiment_runner import CoattentionNetExperimentRunner

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str)
    parser.add_argument('--train_question_path', type=str)
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--num_data_loader_workers', type=int, default=8)
    parser.add_argument('--cache_location', type=str, default="")
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--log_validation', action='store_true')
    parser.add_argument('--exp_name', type=str, default='')
    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                cache_location=args.cache_location,
                                                lr=args.lr,
                                                log_validation=args.log_validation,
                                                exp_name=args.exp_name)
    experiment_runner.train()
