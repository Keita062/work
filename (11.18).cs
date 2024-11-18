using System;

namespace SentakuSortTest
{
    class Program
    {
        static void Main()
        {
            // 乱数生成器の初期化
            Random random = new Random();

            // 配列 a を設定 (0〜99の範囲の10個の整数乱数)
            int[] a = new int[10];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = random.Next(0, 100); // 0〜99の乱数
            }

            // 配列 a の内容を出力
            Console.WriteLine("初期の配列:");
            PrintArray(a);

            // 最小値を求め、a[0] と交換する (選択ソートの初期段階)
            int minIndex = 0;
            for (int i = 1; i < a.Length; i++)
            {
                if (a[i] < a[minIndex])
                {
                    minIndex = i;
                }
            }

            // 最小値と a[0] を交換
            int temp = a[0];
            a[0] = a[minIndex];
            a[minIndex] = temp;

            // 配列 a の内容を出力
            Console.WriteLine("最小値を先頭に移動後の配列:");
            PrintArray(a);
        }

        // 配列を出力するメソッド
        static void PrintArray(int[] array)
        {
            foreach (int value in array)
            {
                Console.Write(value + " ");
            }
            Console.WriteLine();
        }
    }
}

