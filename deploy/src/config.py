middle_prediction = {
    'path_to_model': './src/middle_model/config_middle_predictor/model.tflite',
    'path_to_labels': './src/middle_model/config_middle_predictor/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.2

}

emotion_prediction = {
    'path_to_model': './src/emotion_model/config_emotion_predictor/model.tflite',
    'path_to_labels': './src/emotion_model/config_emotion_predictor/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.2
}