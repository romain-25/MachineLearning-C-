using System;
using System.IO;
using Microsoft.ML;
using System.Collections.Generic;
namespace MachineLearning
{
    class Program
    {
        public static MLContext context;
        private static ITransformer model;
        private static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "train.txt");
        static void Main(string[] args)
        {
            context = new();
            model = GetModel();
            UseModel();
        }

        public static ITransformer BuidAndTrainModel(IDataView data)
        {
            var estimator = context.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                    inputColumnName: nameof(SentimentData.SentimentText))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            Console.WriteLine("Création et entrainement du modèle");
            var model = estimator.Fit(data);
            Console.WriteLine("Fin de l'entrainement");

            return model;
        }

        public static void Evaluate(IDataView data)
        {
            Console.WriteLine("Evaluation de la précision modèle");

            var prédictions = model.Transform(data);
            var metrics = context.BinaryClassification.Evaluate(prédictions);

            Console.WriteLine($"Précision : {metrics.Accuracy:P2}");
            Console.WriteLine("Fin de l'évaluation du modèle");
        }

        public static ITransformer GetModel()
        {
            if (File.Exists("model.zip"))
            {
                return context.Model.Load("model.zip", out DataViewSchema schema);
            }
            var data = context.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, allowQuoting: true);
            var splitDataView = context.Data.TrainTestSplit(data);
            model = BuidAndTrainModel(splitDataView.TestSet);

            Evaluate(data);

            context.Model.Save(model, data.Schema, "model.zip");

            return model;
        }

        public static void UseModel()
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionEngine =
                context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText =
                        "Saison juste complétement nulle, sans histoire, sans détails, sans émotions, une vrai démonstration de réalisateurs incompétents sans livre pour les guider."
                },
                new SentimentData
                {
                    SentimentText =
                        "La pire fin de série !!! Pourquoi gâcher un chef d'oeuvres... en fin pourrie !!! De semaine en semaine les épisodes mon déçus c'était plat et bâclé... la mort de certains personnages son nul et incompréhensible !! Déçu déçu déçu !!!! La pire fin de tout les temps ..."
                },
                new SentimentData
                {
                    SentimentText =
                        "C'est tout simplement la meilleur série de tous les temps, basée sur les romans de George R. R. Martin, les huit saisons sont vraiment géniales. Le spectacle est à couper le souffle. À voir absolument et sans modération..."
                }
            };

            var reviews = context.Data.LoadFromEnumerable(sentiments);
            var prédictions = model.Transform(reviews);
            var prédictionResults =
                context.Data.CreateEnumerable<SentimentPrediction>(prédictions, reuseRowObject: true);

            Console.WriteLine("Test de prédiction");

            foreach (var prédiction in prédictionResults)
            {
                var prédictionText = prédiction.Prediction ? "Critique positive" : "Critique négative";
                Console.WriteLine($"Critique : {prédiction.SentimentText}");
                Console.WriteLine($"Prédiction : {prédictionText}");
                Console.WriteLine($"Probabilité : {prédiction.Probability:P2}");
                Console.WriteLine(
                    "-------------------------------------------------------------------------------------");
            }
        }
    }
}
