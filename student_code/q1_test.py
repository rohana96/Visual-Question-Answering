from external.vqa.vqa import VQA

vqa = VQA('data/mscoco_train2014_annotations.json', 'data/OpenEnded_mscoco_train2014_questions.json')

questions_info_dict = vqa.questions

questions_dict_list = questions_info_dict['questions']
for question_dict in questions_dict_list:
    if question_dict['question_id'] == 2374130:  # 409380: (40500 --> 2374130)
        print(question_dict)

# {'question': 'What position is the man squatting with a glove on playing?', 
# 'image_id': 40938, 
# 'question_id': 409380}

answer_info = vqa.loadQA(409380)
answer_info = vqa.loadQA(2374130)
print(answer_info)
# [{'question_type': 'what', 'multiple_choice_answer': 'catcher', 'answers': [{'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 1}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 2}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 3}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 4}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 5}, {'answer': 'catcher', 'answer_confidence': 'maybe', 'answer_id': 6}, {'answer': 'baseball', 'answer_confidence': 'yes', 'answer_id': 7}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 8}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 9}, {'answer': 'catcher', 'answer_confidence': 'yes', 'answer_id': 10}], 'image_id': 40938, 'answer_type': 'other', 'question_id': 409380}]
