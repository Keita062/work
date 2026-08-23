double teihen, takasa, s;

Console.WriteLine("三角形の面積を求めます");
Console.Write("底辺=");
teihen = double.Parse(Console.ReadLine());
Console.Write("高さ=");
takasa = double.Parse(Console.ReadLine());

s = triangle(teihen, takasa);
Console.Write("三角形の面積=" + s);

double triangle(double teihen, double takasa)
{
    return teihen * takasa / 2;
}

