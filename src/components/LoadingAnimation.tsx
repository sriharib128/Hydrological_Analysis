import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

interface LoadingAnimationProps {
  duration?: number;
}

const steps = [
  { message: 'Training MLP Regressor', duration: 5000 },
  { message: 'Training Random Forest', duration: 5000 },
  { message: 'Training XGBoost', duration: 5000 },
  { message: 'Creating Ensemble Model', duration: 5000 },
  { message: 'Creating Plots', duration: 5000 },
];

export function LoadingAnimation({ duration = 25000 }: LoadingAnimationProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
    }, 5000);

    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + 1;
      });
    }, duration / 100);

    return () => {
      clearInterval(stepInterval);
      clearInterval(progressInterval);
    };
  }, [duration]);

  return (
    <div className="flex flex-col items-center justify-center space-y-6 p-8">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      >
        <Loader2 className="w-12 h-12 text-blue-600" />
      </motion.div>

      <div className="w-full max-w-md space-y-4">
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-800">
            {steps[currentStep].message}
          </p>
          <p className="text-sm text-gray-600">
            Step {currentStep + 1} of {steps.length}
          </p>
        </div>

        <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
          <motion.div
            className="absolute top-0 left-0 h-full bg-blue-600"
            initial={{ width: '0%' }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>

        <div className="flex justify-between text-sm text-gray-600">
          <span>Progress</span>
          <span>{progress}%</span>
        </div>

        <div className="grid grid-cols-5 gap-2">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`h-1 rounded-full ${
                index <= currentStep ? 'bg-blue-600' : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}