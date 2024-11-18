import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Download } from 'lucide-react';
import { motion } from 'framer-motion';
import { Separator } from '@/components/ui/separator';
import { LoadingAnimation } from './LoadingAnimation';
import { FileUpload, FileUploadForm, SubmitButton } from './FileUpload';
import { Button } from './ui/button';

function MLAnalysis() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('train_data', event.target.train_data.files[0]);
    formData.append('test_data', event.target.test_data.files[0]);
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/ml', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
      
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  const groupPlots = (plots) => {
    const eda = plots.slice(0, 3);
    const correlation = plots.slice(3, 5);
    const model = plots.slice(5, 7);
    const test = plots.slice(7);
    return { eda, correlation, model, test };
  };

  if (loading) {
    return <LoadingAnimation />;
  }

  return (
    <div className="p-6 space-y-8">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="max-w-xl mx-auto"
      >
        <FileUploadForm onSubmit={handleSubmit} loading={loading}>
          <FileUpload
            id="train_data"
            name="train_data"
            label="Training Data (CSV)"
            accept=".csv"
          />
          <FileUpload
            id="test_data"
            name="test_data"
            label="Test Data (CSV)"
            accept=".csv"
          />
          <SubmitButton
            loading={loading}
            text="Analyze Data"
            loadingText="Processing Data..."
          />
        </FileUploadForm>
      </motion.div>
      
      {result && (
        <motion.div
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-8"
        >{/* EDA Section */}
          <div>
            <h2 className="text-2xl font-semibold mb-6">Exploratory Data Analysis</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {groupPlots(result.plots).eda.map((plot, index) => (
                <motion.div key={index} variants={itemVariants}>
                  <Card className="p-4 h-full">
                    <h3 className="font-semibold mb-4">{plot.title}</h3>
                    <img
                      src={`data:image/png;base64,${plot.image}`}
                      alt={plot.title}
                      className="rounded-lg shadow-sm w-full"
                    />
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>

          <Separator />

          {/* Correlation Section */}
          <div>
            <h2 className="text-2xl font-semibold mb-6">Correlation Analysis</h2>
            <div className="grid md:grid-cols-2 gap-6">
              {groupPlots(result.plots).correlation.map((plot, index) => (
                <motion.div key={index} variants={itemVariants}>
                  <Card className="p-4 h-full">
                    <h3 className="font-semibold mb-4">{plot.title}</h3>
                    <img
                      src={`data:image/png;base64,${plot.image}`}
                      alt={plot.title}
                      className="rounded-lg shadow-sm w-full"
                    />
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>

          <Separator />

          {/* Model Comparison Section */}
          <div>
            <h2 className="text-2xl font-semibold mb-6">Model Comparison</h2>
            <div className="grid md:grid-cols-2 gap-6">
              {groupPlots(result.plots).model.map((plot, index) => (
                <motion.div key={index} variants={itemVariants}>
                  <Card className="p-4 h-full">
                    <h3 className="font-semibold mb-4">{plot.title}</h3>
                    <img
                      src={`data:image/png;base64,${plot.image}`}
                      alt={plot.title}
                      className="rounded-lg shadow-sm w-full"
                    />
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>

          <Separator />

          {/* Test Results Section */}
          <div>
            <h2 className="text-2xl font-semibold mb-6">Test Results</h2>
            <div className="grid gap-6">
              <motion.div variants={itemVariants}>
                <Card className="p-6">
                  <h3 className="font-semibold mb-4">Test Results Plot</h3>
                  <img
                    src={`data:image/png;base64,${groupPlots(result.plots).test[0].image}`}
                    alt="Test Results"
                    className="rounded-lg shadow-sm w-full max-w-[50%] mx-auto"
                  />
                </Card>
              </motion.div>

              <motion.div variants={itemVariants}>
                <div className="grid md:grid-cols-4 gap-4">
                  {result.metrics.final_results.map((finalResult, index) => (
                    <Card key={index} className="p-6">
                      <h3 className="font-semibold text-lg mb-4">{finalResult.model}</h3>
                      <div className="space-y-4">
                        <div>
                          <p className="text-sm text-gray-600">MAE</p>
                          <p className="text-2xl font-bold text-blue-600">
                            {finalResult.mae.toFixed(5)}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">R²</p>
                          <p className="text-2xl font-bold text-green-600">
                            {finalResult.r2.toFixed(5)}
                          </p>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              </motion.div>

              <motion.div
                variants={itemVariants}
                className="flex justify-center"
              >
                <Button
                  asChild
                  className="bg-green-600 hover:bg-green-700 transition-all transform hover:scale-105"
                >
                  <a
                    href={`data:text/csv;base64,${result.predictions_csv_base64}`}
                    download="predictions.csv"
                    className="flex items-center"
                  >
                    <Download className="mr-2" />
                    Download Predictions CSV
                  </a>
                </Button>
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default MLAnalysis;