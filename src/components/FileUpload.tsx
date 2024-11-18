import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';

interface FileUploadProps {
  id: string;
  label: string;
  accept?: string;
  required?: boolean;
  name: string;
  className?: string;
}

export function FileUpload({
  id,
  label,
  accept,
  required = true,
  name,
  className = '',
}: FileUploadProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`space-y-2 ${className}`}
    >
      <Label htmlFor={id} className="text-sm font-medium text-gray-700">
        {label}
      </Label>
      <div className="relative">
        <Input
          id={id}
          type="file"
          name={name}
          accept={accept}
          required={required}
          className="cursor-pointer file:mr-4 file:py-2 file:px-4 
                   file:rounded-full file:border-0 file:text-sm file:font-semibold
                   file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100
                   focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        />
        <div className="absolute inset-0 pointer-events-none border-2 border-dashed border-gray-200 rounded-lg" />
      </div>
    </motion.div>
  );
}

interface SubmitButtonProps {
  loading: boolean;
  text: string;
  loadingText?: string;
  className?: string;
}

export function SubmitButton({
  loading,
  text,
  loadingText = 'Processing...',
  className = '',
}: SubmitButtonProps) {
  return (
    <Button
      type="submit"
      disabled={loading}
      className={`w-full bg-blue-600 hover:bg-blue-700 transition-all
                transform hover:scale-105 disabled:opacity-50 
                disabled:cursor-not-allowed ${className}`}
    >
      {loading ? (
        <span className="flex items-center justify-center">
          <Upload className="animate-bounce mr-2" />
          {loadingText}
        </span>
      ) : (
        text
      )}
    </Button>
  );
}

interface FileUploadFormProps {
  onSubmit: (event: React.FormEvent) => Promise<void>;
  loading: boolean;
  children: React.ReactNode;
  className?: string;
}

export function FileUploadForm({
  onSubmit,
  loading,
  children,
  className = '',
}: FileUploadFormProps) {
  return (
    <Card className={`p-6 shadow-lg ${className}`}>
      <form onSubmit={onSubmit} className="space-y-6">
        {children}
      </form>
    </Card>
  );
}