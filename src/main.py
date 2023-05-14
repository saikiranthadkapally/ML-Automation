from utils import MyHyperModel,kt,prepare_dataset
from config import *
def tune_train_model():
    
    hypermodel = MyHyperModel(num_classes=N_LABELS)
    
    directory = "./"
    tuner = kt.Hyperband(hypermodel,
                         objective='val_loss',
                         max_epochs=10,
                         factor=3,
                         seed=123,
                         directory=directory,
                         project_name='experminets')
    
    x_data,y_data = prepare_dataset()

    tuner.search(x_data,y_data,epochs=50,steps_per_epoch=90,validation_split=0.2)
    
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    
    model = tuner.hypermodel.build(best_hps)

    model.summary()
    
    history = model.fit(x_data,y_data, epochs=MODEL_CONF['EPOCHS'], validation_split=0.2,
                        steps_per_epoch=90)

    
    print(history)
    model.save(OUTPUT_FOLDER+'/model.h5')

if __name__ == "__main__":
    tune_train_model()