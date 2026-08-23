double x;

Console.WriteLine("実数を入力してください");
Console.Write("x=");
x = double.Parse(Console.ReadLine());

Console.WriteLine("Sqrt(x)=" + Math.Sqrt(x));
Console.WriteLine("Exp(x)=" + Math.Exp(x));
Console.WriteLine("Log(x)=" + Math.Log(x));
Console.WriteLine("Log10(x)=" + Math.Log10(x));
