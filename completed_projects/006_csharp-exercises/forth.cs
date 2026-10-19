double shincho;
double taiju;
double bmi;

Console.Write("身長[cm]=");
shincho = double.Parse(Console.ReadLine());
Console.Write("体重[kg]=");
taiju = double.Parse(Console.ReadLine());

shincho = shincho / 100.0;
bmi = taiju / (shincho * shincho);
Console.Write("BMI=");
Console.WriteLine(bmi);
if (bmi >= 25)
    Console.WriteLine("肥満です");
else
    Console.WriteLine("肥満ではありません");
