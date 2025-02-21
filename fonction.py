from pytorch_forecasting import TimeSeriesDataSet
import pickle
import csv
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import os
import torch
import pandas as pd

class TFTWrapper(pl.LightningModule):
    def __init__(self, train_dataset):
        super().__init__()
        self.tft_model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=0.001,
            hidden_size=160,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=160,
            output_size=7,  # nombre de quantiles
            loss=QuantileLoss(),
            log_interval=10, 
            reduce_on_plateau_patience=4,
        )
    
    def forward(self, batch):
        # Déballer explicitement le batch : x est le dictionnaire attendu
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        output = self.tft_model(x)
        # Si output est un tuple, on prend le premier élément
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output
    
    def training_step(self, batch, batch_idx):
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        output = self.tft_model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = self.tft_model.loss(output, x["decoder_target"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        output = self.tft_model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = self.tft_model.loss(output, x["decoder_target"])
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return self.tft_model.configure_optimizers()
    
    # Ajout d'une méthode predict déléguée à self.tft_model
    def predict(self, dataloader, **kwargs):
        return self.tft_model.predict(dataloader, **kwargs)
    
    # Ajout d'une méthode to_prediction_dataframe déléguée à self.tft_model
    def to_prediction_dataframe(self, x, raw_predictions):
        return self.tft_model.to_prediction_dataframe(x, raw_predictions)

def transforme_data_selected_for_TFT(df_group):

    # Créer le dataset de test/validation
    prediction_dataset = TimeSeriesDataSet(
        df_group,
        group_ids=["group"],
        target="total_guests",
        time_idx="time_idx",
        min_encoder_length=7,
        max_encoder_length=7,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=["total_guests"],
        static_categoricals=["lounge_name"],
        time_varying_known_reals=["DOW_cos", "DOW_sin","WOY_cos","WOY_sin"],
        target_normalizer=None
    )
    
    prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=64, num_workers=7)

    #Batch liste: 
    batch_list = []
    for batch in prediction_dataloader:
        batch_list.append(batch)

    with open("saved_prediction_dataloader.pkl", "wb") as f:
        pickle.dump(batch_list, f)

    return prediction_dataloader

def prediction(prediction_dataloader):

    with open("train_dataset_template.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    

    # Obtenir le répertoire du fichier courant
    base_dir = os.path.dirname(os.path.abspath(__file__))


    # Lire le chemin du checkpoint depuis le CSV
    with open("checkpoint_path.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            checkpoint_path = row["checkpoint_path"]
            checkpoint_path = os.path.join(base_dir, "..", checkpoint_path)
            checkpoint_path = os.path.abspath(checkpoint_path)
            break  # On prend la première ligne

    # Charger le modèle
    best_model = TFTWrapper.load_from_checkpoint(checkpoint_path, train_dataset=train_dataset)

    best_model.to("cpu")

    predictions = best_model.predict(prediction_dataloader, mode="raw",return_x=True)

    raw_predictions = predictions.output.prediction  # tenseur des prédictions
    x = predictions.x  # dictionnaire contenant les infos d'entrée

    return raw_predictions, x


def custom_to_prediction_dataframe(x, raw_predictions):
    """
    Construit un DataFrame à partir du dictionnaire d'entrée `x` et du tenseur `raw_predictions`.
    
    On suppose que `x` contient notamment :
      - "groups" : l'identifiant du groupe
      - "decoder_time_idx" : l'index temporel pour la prédiction
      - "decoder_target" : la valeur réelle (target)
    
    Le tenseur raw_predictions est supposé avoir la forme 
      (n_samples, prediction_length, quantile_count)
    et on choisit ici le quantile médian pour la prédiction (ou le seul quantile s'il n'y en a qu'un).
    """
    # Convertir les éléments de x en numpy s'ils sont des tenseurs
    groups = x["groups"]
    if isinstance(groups, torch.Tensor):
        groups = groups.cpu().numpy().flatten()
    
    time_idx = x["decoder_time_idx"]
    if isinstance(time_idx, torch.Tensor):
        time_idx = time_idx.cpu().numpy().flatten()
    
    decoder_target = x["decoder_target"]
    if isinstance(decoder_target, torch.Tensor):
        decoder_target = decoder_target.cpu().numpy().flatten()
    
    # Convertir raw_predictions en numpy
    preds = raw_predictions
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    
    # Supposons que preds est de forme (n_samples, prediction_length, quantile_count)
    # On prend la première valeur dans la dimension prediction_length et le quantile médian (si plusieurs quantiles)
    if preds.shape[2] > 1:
        median_index = preds.shape[2] // 2
        pred_values = preds[:, 0, median_index]
    else:
        pred_values = preds[:, 0, 0]
    
    # Créer le DataFrame
    df = pd.DataFrame({
        "group": groups,
        "time_idx": time_idx,
        "total_guest_real": decoder_target,
        "prediction": pred_values
    })
    
    return df








