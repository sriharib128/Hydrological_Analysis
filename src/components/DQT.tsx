import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { LoadingAnimation } from './LoadingAnimation';
import { FileUpload, FileUploadForm, SubmitButton } from './FileUpload';

function DQT() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file', event.target.water_quality.files[0]);
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/dqt', {
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

  // if (loading) {
  //   return <LoadingAnimation />;
  // }

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
            id="water_quality"
            name="water_quality"
            label="Input Data"
            accept=".csv"
          />
          <SubmitButton
            loading={loading}
            text="Analyze Water Quality"
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
        >
          <div className="grid md:grid-cols-3 gap-6">
            <motion.div variants={itemVariants}>
              <Card className="p-4 h-full">
                <h3 className="font-semibold mb-4">Release Plot</h3>
                <img
                  src={result.release_plot}
                  alt="Release Plot"
                  className="rounded-lg shadow-sm w-full"
                />
              </Card>
            </motion.div>
            <motion.div variants={itemVariants}>
              <Card className="p-4 h-full">
                <h3 className="font-semibold mb-4">Flow Moving Average</h3>
                <img
                  src={result.flow_moving_avg_plot}
                  alt="Flow Moving Average Plot"
                  className="rounded-lg shadow-sm w-full"
                />
              </Card>
            </motion.div>
            <motion.div variants={itemVariants}>
              <Card className="p-4 h-full">
                <h3 className="font-semibold mb-4">Frequency Curve</h3>
                <img
                  src={result.frequency_curve_plot}
                  alt="Frequency Curve Plot"
                  className="rounded-lg shadow-sm w-full"
                />
              </Card>
            </motion.div>
          </div>

          <motion.div
            variants={itemVariants}
            className="flex justify-center"
          >
            <Card className="p-6 max-w-sm w-full">
              <h3 className="font-semibold text-lg mb-4 text-center">7Q5 Value</h3>
              <p className="text-4xl font-bold text-blue-600 text-center">
                {parseFloat(result['7Q5_m3_s']).toFixed(5)} m³/s
              </p>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}

export default DQT;