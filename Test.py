import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions



# Initial Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(r'C:/Users/admin/Downloads/plantdisease_dataset')
set_printoptions(precision=2, suppress=True)

# Loading Model
my_model = load_model('plant_cnn_0.94.h5')
print(my_model.summary(), '\n')

# Parameters: Weights and Biases
print('Plant CNN last layer bias:')
print(my_model.get_weights()[-1])
print('Plant CNN last layer weights:')
print(my_model.get_weights()[-2])

# Evaluation test\n",
eval_idg = ImageDataGenerator(rescale=1. / 255.)
eval_g = eval_idg.flow_from_directory(directory=r'C:/Users/admin/Downloads/plantdisease_dataset/Testing/',
                                      target_size=(100, 100),
                                      class_mode='binary',
                                      batch_size=125,
                                      shuffle=False)
(eval_loss, eval_acc) = my_model.evaluate_generator(generator=eval_g, steps=1)
print('evaluation Loss over never-before-seen images is: {:.4f}'.format(eval_loss))
print('evaluation Accuracy over never-before-seen images is: {:4.2f}%'.format(eval_acc*100), '\n')

# Individual Predictions
pred_idg = eval_idg
pred_g = eval_g
pred = my_model.predict_generator(generator=pred_g, steps=1)
print(pred_g.filenames, '\n')
print(pred_g.class_indices, '\n')
print(pred[0:5], '\n')

#print(pred[30:60], '\n')
#print(pred[60:90] ,'\n')
#print(pred[90:120] , '\n')
#print(pred[120:150] , '\n')
#print(pred[150:180] , '\n')
#print(pred[180:210] , '\n')
#print(pred[210:240] , '\n')
#print(pred[240:270] , '\n')
#print(pred[270:300], '\n')
#print(pred[300:330] , '\n')
#print(pred[330:360], '\n')
#print(pred[360:390], '\n')
#print(pred[390:420], '\n')
#print(pred[420:450], '\n')
#print(pred[450:480], '\n')
#print(pred[480:510], '\n')
#print(pred[510:540], '\n')
#print(pred[540:570], '\n')
#print(pred[570:600], '\n')
#print(pred[600:630], '\n')
#print(pred[630:660], '\n')
#print(pred[660:690], '\n')
#print(pred[690:720], '\n')
#print(pred[720:750], '\n')
#print(pred[750:780], '\n')
#print(pred[780:810], '\n')
#print(pred[810:840], '\n')
#print(pred[840:870], '\n')
#print(pred[870:900], '\n')
#print(pred[900:930], '\n')
#print(pred[930:960], '\n')
#print(pred[960:990], '\n')
#print(pred[990:1020], '\n')
#print(pred[1020:1050], '\n')
#print(pred[1050:1080], '\n')
#print(pred[1080:1100], '\n')
#print(pred[1110:1140], '\n')





