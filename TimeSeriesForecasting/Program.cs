using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;

namespace TimeSeriesForecasting
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<EnergyData>(@"Data\energy_hourly.csv", separatorChar: ',', hasHeader: true);

            var pipeline = context.Forecasting.ForecastBySsa(
                "Forecast",
                nameof(EnergyData.Energy),
                windowSize: 5,
                seriesLength: 10,
                trainSize: 100,
                horizon: 4);

            var model = pipeline.Fit(data);

            var forecastingEngine = model.CreateTimeSeriesEngine<EnergyData, EnergyForecast>(context);

            var forecasts = forecastingEngine.Predict();


            foreach (var item in forecasts.Forecast)
            {
                Console.WriteLine($"{item}");
            }

        }
    }
}
