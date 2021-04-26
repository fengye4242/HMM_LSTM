# params for training network
manual_seed = None
batch_size = 200
num_epochs=1
save_step_pre=10
# params for source dataset
src_dataset = 'data/subject_1.mat'
src_encoder_restore = "snapshots/ADDA-source-encoder-3fc-final-1.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-3fc-final-1.pt"
src_model_trained = True