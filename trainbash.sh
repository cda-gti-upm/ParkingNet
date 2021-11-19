 #!/bin/bash$

#screen -S jorgecerve
#screen -r jorgecerve

#conda activate tf-gpu 


# Dataset ETSIT
# Train
#python src/train.py --data_route=/media/Data/jcp/ETSIT --conv_base=Resnet50 --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/ETSIT --conv_base=Xception --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/ETSIT --conv_base=NASNetLarge --nb_epoch=50 --batch_size=16 --learning_rate=1e-4

# Test
#python src/test.py --data_route=/media/Data/jcp/ETSIT --conv_base=Resnet50 --batch_size=16 --checkpoint_name=Resnet50_merged_data_2021Nov07_22h38m31s


#PKLOT DATASET PUCPR
#python src/train.py --data_route=/media/Data/jcp/PUCPR --conv_base=Resnet50 --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/PUCPR --conv_base=Xception --nb_epoch=50 --batch_size=16 --learning_rate=1e-4 
#python src/train.py --data_route=/media/Data/jcp/PUCPR --conv_base=NASNetLarge --nb_epoch=50 --batch_size=16 --learning_rate=1e-4  

#PKLOT DATASET UFPR04
#python src/train.py --data_route=/media/Data/jcp/UFPR04 --conv_base=Resnet50 --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/UFPR04 --conv_base=Xception --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/UFPR04 --conv_base=NASNetLarge --nb_epoch=50 --batch_size=16 --learning_rate=1e-4

#PKLOT DATASET UFPR05
#python src/train.py --data_route=/media/Data/jcp/UFPR05 --conv_base=Resnet50 --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/UFPR05 --conv_base=Xception --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python src/train.py --data_route=/media/Data/jcp/UFPR05 --conv_base=NASNetLarge --nb_epoch=50 --batch_size=16 --learning_rate=1e-4



# Predict
python src/predict.py --data_route=/media/Data/jcp/ETSIT --conv_base=Resnet50 --checkpoint_name=Resnet50_ETSIT_2021Nov13_01h30m57s
python src/predict.py --data_route=/media/Data/jcp/ETSIT --conv_base=Xception --checkpoint_name=Xception_ETSIT_2021Nov09_20h09m09s
python src/predict.py --data_route=/media/Data/jcp/ETSIT --conv_base=NASNetLarge --checkpoint_name=NASNetLarge_ETSIT_2021Nov10_01h46m37s

python src/predict.py --data_route=/media/Data/jcp/PUCPR --conv_base=Resnet50 --checkpoint_name=Resnet50_PUCPR_2021Nov13_04h52m31s
python src/predict.py --data_route=/media/Data/jcp/PUCPR --conv_base=Xception --checkpoint_name=Xception_PUCPR_2021Nov10_18h59m49s
python src/predict.py --data_route=/media/Data/jcp/PUCPR --conv_base=NASNetLarge --checkpoint_name=NASNetLarge_PUCPR_2021Nov10_20h01m39s

python src/predict.py --data_route=/media/Data/jcp/UFPR04 --conv_base=Resnet50 --checkpoint_name=Resnet50_UFPR04_2021Nov13_05h46m21s
python src/predict.py --data_route=/media/Data/jcp/UFPR04 --conv_base=Xception --checkpoint_name=Xception_UFPR04_2021Nov10_22h49m20s
python src/predict.py --data_route=/media/Data/jcp/UFPR04 --conv_base=NASNetLarge --checkpoint_name=NASNetLarge_UFPR04_2021Nov11_00h17m38s

python src/predict.py --data_route=/media/Data/jcp/UFPR05 --conv_base=Resnet50 --checkpoint_name=Resnet50_UFPR05_2021Nov13_07h02m26s
python src/predict.py --data_route=/media/Data/jcp/UFPR05 --conv_base=Xception --checkpoint_name=Xception_UFPR05_2021Nov11_04h34m13s
python src/predict.py --data_route=/media/Data/jcp/UFPR05 --conv_base=NASNetLarge --checkpoint_name=NASNetLarge_UFPR05_2021Nov11_05h24m53s



#tensorboard dev upload --logdir 'logs/'