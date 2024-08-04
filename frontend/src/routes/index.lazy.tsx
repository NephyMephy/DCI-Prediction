import { createLazyFileRoute } from "@tanstack/react-router";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Area, AreaChart, CartesianGrid, XAxis } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

export const Route = createLazyFileRoute("/")({
  component: Index,
});

function Index() {
  const [date, setDate] = useState<string>("");
  const [corps, setCorps] = useState<string>("");
  const [open, setOpen] = useState(false);
  const [prediction, setPrediction] = useState<any>([]);
  const [mlPrediction, setMLPrediction] = useState<any>([]);
  const [copyprediction, setcopyPrediction] = useState<any>([]);
  const [previousScores, setPreviousScores] = useState<any>([]);
  const [years, setYears] = useState<string[]>([]);
  const [selectedYear, setSelectedYear] = useState<string>("");
  const handlePredict = async () => {
    const formdata = new FormData();
    formdata.append("date[]", date);
    formdata.append("corps[]", corps);

    const res = await fetch(import.meta.env.VITE_DCI_URL, {
      method: "POST",
      body: formdata,
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
    });
    const data = await res.json();
    setMLPrediction(data.predictions[0]);
    const filtered = (data.previous[0] as any[]).map((x) => ({
      score: x.Score,
      date: x.Date,
    }));
    filtered.push({
      score: data.predictions[0][2],
      date: data.predictions[0][1],
    });

    const insertYear: string[] = [];
    filtered.map((x) => {
      if (!insertYear.includes(new Date(x.date).getFullYear().toString())) {
        insertYear.push(new Date(x.date).getFullYear().toString());
      }
    });

    setPrediction(filtered);
    setYears(insertYear);
    // console.log("Predictions array", prediction);
    setOpen(true);
    console.log("Predictions", filtered);
  };

  useEffect(() => {
    const current: any = [];
    const prev: any = [];
    console.log(mlPrediction);
    prediction.map((x: any) => {
      if (x.date.includes(mlPrediction[1].split("-")[0])) {
        current.push({
          date: new Date(x.date).toISOString().split("T")[0],
          score: x.score,
        });
      }
      if (x.date.includes(Number(selectedYear))) {
        prev.push({
          date: new Date(x.date).toISOString().split("T")[0],
          score: x.score,
        });
      }
    });

    setPreviousScores(prev);
    setcopyPrediction(current);
    console.log(previousScores, prediction);
  }, [selectedYear]);

  const chartConfig = {
    previous: {
      label: "Previous",
      color: "hsl(var(--chart-1))",
    },
    current: {
      label: "Current",
      color: "hsl(var(--chart-2))",
    },
  } satisfies ChartConfig;
  return (
    <div className="flex items-center justify-center min-h-screen w-full ">
      <Card className="min-w-96 max-w-96 flex items-center flex-col justify-center text-center">
        <CardHeader>
          <CardTitle>DCI Prediction</CardTitle>
          <CardDescription className="text-xs text-left">
            A machine learning model to predict future Drum Corps International
            (DCI) scores based on historical data
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col space-y-3">
          <Input
            placeholder="Enter date"
            name="date[]"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="outline-none focus:outline-none"
          />
          <Input
            placeholder="Corps Name"
            name="corps[]"
            value={corps}
            onChange={(e) => setCorps(e.target.value)}
            className="outline-none focus:outline-none"
          />
          <Button type="submit" variant={"default"} onClick={handlePredict}>
            Predict Scores
          </Button>
        </CardContent>
      </Card>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="min-w-[30rem]">
          <DialogHeader>
            <DialogTitle className="flex space-x-4 items-center">
              <span>DCI Prediction Results</span>
              <Select
                defaultValue={years[years.length - 2]}
                onValueChange={setSelectedYear}
              >
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Date range" />
                </SelectTrigger>
                <SelectContent>
                  {years.map((year) => (
                    <SelectItem value={year} key={year}>
                      {year}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </DialogTitle>
            <DialogDescription className="flex flex-col">
              <Card>
                <CardHeader>
                  <CardTitle>Area Chart</CardTitle>
                  <CardDescription></CardDescription>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={chartConfig}>
                    <AreaChart
                      accessibilityLayer
                      data={previousScores}
                      margin={{
                        left: 12,
                        right: 12,
                      }}
                    >
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="date"
                        tickLine={false}
                        axisLine={false}
                        tickMargin={2}
                        tickFormatter={(value) => {
                          return new Date(value).toLocaleDateString("en-US", {
                            month: "short",
                          });
                        }}
                      />
                      <ChartTooltip
                        cursor={false}
                        content={<ChartTooltipContent indicator="line" />}
                      />
                      <Area
                        dataKey="score"
                        type="natural"
                        fill="var(--color-current)"
                        fillOpacity={0.4}
                        stroke="var(--color-current)"
                      />
                    </AreaChart>
                  </ChartContainer>
                  <ChartContainer config={chartConfig}>
                    <AreaChart
                      accessibilityLayer
                      data={copyprediction}
                      margin={{
                        left: 12,
                        right: 12,
                      }}
                    >
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="date"
                        tickLine={false}
                        axisLine={false}
                        tickMargin={8}
                        tickFormatter={(value) => {
                          return new Date(value).toLocaleDateString("en-US", {
                            month: "short",
                          });
                        }}
                      />
                      <ChartTooltip
                        cursor={false}
                        content={<ChartTooltipContent indicator="line" />}
                      />
                      <Area
                        dataKey="score"
                        type="natural"
                        fill="var(--color-current)"
                        fillOpacity={0.4}
                        stroke="var(--color-current)"
                      />
                    </AreaChart>
                  </ChartContainer>
                </CardContent>
                <CardFooter>
                  <div className="flex w-full items-start gap-2 text-sm">
                    <div className="grid gap-2">
                      <div className="flex items-center gap-2 font-medium leading-none">
                        Trending up by 5.2% this month{" "}
                      </div>
                      <div className="flex items-center gap-2 leading-none text-muted-foreground">
                        January - June 2024
                      </div>
                    </div>
                  </div>
                </CardFooter>
              </Card>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
}
