import PageHeader from "../components/PageHeader";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../components/ui/tabs";
import TrainForm from "../features/training/TrainForm";
import RunsTable from "../features/training/RunsTable";
import ModelsTable from "../features/training/ModelsTable";

export default function TrainingPage() {
  return (
    <div>
      <PageHeader title="Training" subtitle="Train, monitor runs, manage registered models" />
      <Tabs defaultValue="train">
        <TabsList>
          <TabsTrigger value="train">Train</TabsTrigger>
          <TabsTrigger value="runs">Runs</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>
        <TabsContent value="train"><TrainForm /></TabsContent>
        <TabsContent value="runs"><RunsTable /></TabsContent>
        <TabsContent value="models"><ModelsTable /></TabsContent>
      </Tabs>
    </div>
  );
}
