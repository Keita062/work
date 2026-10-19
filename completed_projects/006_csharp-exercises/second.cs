double shinchou;
double taijuu;
double bmi;

shinchou = 170.5; 
taijuu = 57.6; 

bmi = taijuu / Math.Pow(shinchou / 100, 2);

Console.Write("BMI=");
Console.WriteLine(bmi);

