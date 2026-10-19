double hankei;
double menseki;

Console.Write("半径＝");
hankei = double.Parse(Console.ReadLine());

menseki = Math.PI * Math.Pow(hankei, 2);

Console.WriteLine("面積＝" + menseki);
