using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

        public string Sentiment
        {
            get
            {
                string _result = "";

                if(Probability < .5)
                {
                    _result = "Negative";
                }
                else if (Probability >= .5 && Probability <= .7)
                {
                    _result = "Neutral";
                }
                else
                {
                    _result = "Positive";
                }

                return _result;
            }
        }
    }
}