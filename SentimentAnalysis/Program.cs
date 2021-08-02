using Microsoft.ML;
using System;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<SentimentData>(@"Data\stock_data.csv", hasHeader: true, separatorChar: ',');

            var pipeline = context.Transforms.Expression("Label", "(x) => x == 1 ? true : false", "Sentiment")
                .Append(context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text)))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var _obj = new SentimentData
            {
                Text = "I would buy MSFT shares"
            };

            var prediction = predictionEngine.Predict(_obj);

            Console.WriteLine($"Prediction {prediction.Prediction} With Probability {prediction.Probability} ({prediction.Sentiment})");
        }
    }
}
