import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Separator } from '@/components/ui/separator';
import { LoadingAnimation } from './LoadingAnimation';
import { FileUpload, FileUploadForm, SubmitButton } from './FileUpload';

function Empirical() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('runoff_tif_path', event.target.runoff.files[0]);
    formData.append('rainfall_tif_path', event.target.rainfall.files[0]);
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/empirical', {
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
            id="runoff"
            name="runoff"
            label="Runoff TIF File"
            accept=".tif"
          />
          <FileUpload
            id="rainfall"
            name="rainfall"
            label="Rainfall TIF File"
            accept=".tif"
          />
          <SubmitButton
            loading={loading}
            text="Analyze Files"
            loadingText="Processing Files..."
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
          <div>
            <h2 className="text-2xl font-semibold mb-6">Input Analysis</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <motion.div variants={itemVariants}>
                <Card className="p-4 h-full">
                  <h3 className="font-semibold mb-4">Runoff Coefficient</h3>
                  <img
                    src={result.plots.runoff_coeff}
                    alt="Runoff Coefficient"
                    className="rounded-lg shadow-sm w-full"
                  />
                </Card>
              </motion.div>
              <motion.div variants={itemVariants}>
                <Card className="p-4 h-full">
                  <h3 className="font-semibold mb-4">Rainfall Intensity</h3>
                  <img
                    src={result.plots.rainfall}
                    alt="Rainfall Intensity"
                    className="rounded-lg shadow-sm w-full"
                  />
                </Card>
              </motion.div>
            </div>
          </div>

          <Separator className="my-8" />

          <div>
            <h2 className="text-2xl font-semibold mb-6">Output Analysis</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <motion.div variants={itemVariants} className="md:col-span-2">
                <Card className="p-4 h-full">
                  <h3 className="font-semibold mb-4">Discharge Map</h3>
                  <img
                    src={result.plots.discharge_map}
                    alt="Discharge Map"
                    className="rounded-lg shadow-sm w-full"
                  />
                </Card>
              </motion.div>
              <motion.div variants={itemVariants} className="space-y-6">
                <Card className="p-6">
                  <h3 className="font-semibold text-lg mb-2">Total Discharge</h3>
                  <p className="text-3xl font-bold text-blue-600">
                    {parseFloat(result.values.total_discharge).toFixed(5)}
                  </p>
                </Card>
                <Card className="p-6">
                  <h3 className="font-semibold text-lg mb-2">Reservoir Area</h3>
                  <p className="text-3xl font-bold text-blue-600">
                    {parseFloat(result.values.reservoir_area).toFixed(5)}
                  </p>
                </Card>
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default Empirical;