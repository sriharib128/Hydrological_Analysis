import React, { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { Droplet, Brain, TestTube2 } from 'lucide-react';
import Empirical from '@/components/Empirical';
import MLAnalysis from '@/components/MLAnalysis';
import DQT from '@/components/DQT';

function App() {
  const [activeTab, setActiveTab] = useState('empirical');

  return (
    <div className="w-screen min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto min-h-full flex flex-col">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Water Analysis Dashboard</h1>
          <p className="text-gray-600">Comprehensive water analysis tools for empirical, machine learning, and quality testing</p>
        </header>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4 flex-grow flex flex-col">
          <TabsList className="grid grid-cols-3 gap-4 bg-white/50 p-1 rounded-lg">
            <TabsTrigger
              value="empirical"
              className="data-[state=active]:bg-white data-[state=active]:shadow-md transition-all"
            >
              <Droplet className="w-4 h-4 mr-2" />
              Empirical Analysis
            </TabsTrigger>
            <TabsTrigger
              value="ml"
              className="data-[state=active]:bg-white data-[state=active]:shadow-md transition-all"
            >
              <Brain className="w-4 h-4 mr-2" />
              ML Analysis
            </TabsTrigger>
            <TabsTrigger
              value="dqt"
              className="data-[state=active]:bg-white data-[state=active]:shadow-md transition-all"
            >
              <TestTube2 className="w-4 h-4 mr-2" />
              DQT
            </TabsTrigger>
          </TabsList>

          {/* Tab Content Area */}
          <Card className="border-none shadow-lg bg-white/80 backdrop-blur-sm flex-grow">
            <div className="bg-white min-h-screen p-4 overflow-y-auto">
              <TabsContent value="empirical" className="m-0 space-y-4">
                <Empirical />
              </TabsContent>
              <TabsContent value="ml" className="m-0 space-y-4">
                <MLAnalysis />
              </TabsContent>
              <TabsContent value="dqt" className="m-0 space-y-4">
                <DQT />
              </TabsContent>
            </div>
          </Card>
        </Tabs>
      </div>
    </div>
  );
}

export default App;
