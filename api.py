import math

def T(k):
    if k == 1:
        return 1
    else:
        return T(k // 2) + T(k - k // 2) + 2 * k - 1

def calculate_correction(K):
    x = 0
    while 2 ** (x) < K:  # 找到满足条件的 x
        x += 1
    return x

def main():
    # 用户输入正整数 n 和 m
    n = int(input("请输入正整数 n: "))
    m = int(input("请输入正整数 m: "))
    
    # 计算 K
    K = math.ceil(n / m)
    
    # 计算余数
    remainder = n % m
    
    # 如果余数为正且不大于 m // 2，则需要计算修正量 x
    if remainder <= m // 2 and remainder > 0:
        correction = calculate_correction(K)
        result = T(K) - correction
    else:
        result = T(K)
    
    # 输出结果
    print(f"T({K}) 的结果是: {result}")

if __name__ == "__main__":
    main()