## Pythonの高速化
<details>
<summary><b>入力受付の高速化</b></summary>

```python
import sys
input = sys.stdin.readline
```
</details>

<details>
<summary><b>PyPy-再帰高速</b></summary>

```python
import pypyjit
pypyjit.set_param('max_unroll_recursion=-1')
```
</details>

## 便利

<details>
<summary><b>大文字小文字</b></summary>

#### 変換
```python
str = "abcXYZ"
str.upper()
str.lower()
```
#### 判定
```python
str.isupper()
str.islower()
```
</details>

<details>
<summary><b>dict型のソート</b></summary>

#### 変換
```python
d = {"banana": 3, "apple": 1, "orange": 2}

# keyで並び替え
d_sort_key = sorted(d.items(), key = lambda x: x[0])
# [("apple", 1), ("banana", 3), ("orange", 2)]

# valueで並び替え
d_sort_val = sorted(d.items(), key = lambda x: x[1])
# [("apple", 1), ("orange", 2), ("banana", 3)]

```
</details>

<details>
<summary><b>Counterの使い方</b></summary>

```python
from collections import Counter

moji1 = "apple"
cnt1 = Counter(moji1)

moji2 = "average"
cnt2 = Counter(moji2)

print(cnt2 - cnt1)
# Counter({"a": 1, "v": 1, "e": 1, "r": 1, "g": 1})

```
</details>

## 数学　高速

<details>
<summary><b>素数</b></summary>

Nまでの素数を算出
```python
def primeNumber(N):
    N = int(N ** (1 / 2) + 0.5) + 1
    primeList = [True] * (N + 1)
    primeList[0] = False
    primeList[1] = False
    
    for i in range(2, N + 1):
        if primeList[i]:
            for j in range(2 * i, N + 1, i):
                primeList[j] = False
    
    prime = []
    
    for i in range(2, N + 1):
        if primeList[i]:
            prime.append(i)
    
    return prime

N = int(input())

prime = primeNumber(N)
```
</details>

<details>
<summary><b>素因数分解</b></summary>

```python
def primeNumber(N):
    N = int(N ** (1 / 2) + 0.5) + 1
    primeList = [True] * (N + 1)
    primeList[0] = False
    primeList[1] = False
    
    for i in range(2, N + 1):
        if primeList[i]:
            for j in range(2 * i, N + 1, i):
                primeList[j] = False
    
    prime = []
    
    for i in range(2, N + 1):
        if primeList[i]:
            prime.append(i)
    
    return prime

def factorization(n, prime):
    ret = []
    tmp = n
    for i in prime:
        if tmp % i == 0:
            cnt = 0
            while tmp % i == 0:
                cnt += 1
                tmp //= i
            ret.append([i, cnt])
    
    if tmp != 1:
        ret.append([tmp, 1])
    
    if ret == []:
        ret.append([n, 1])
    
    return ret

N = int(input())

prime = primeNumber(N)
fac = factorization(N, prime)
```
</details>

## アルゴリズム及びデータ構造

<details>
<summary><b>Union Find</b></summary>

```python
import sys
sys.setrecursionlimit(10 ** 8)

def root(x):
    if par[x] < 0:
        return x
    else:
        par[x] = root(par[x])
        return par[x]
 
def union(x, y):
    x = root(x)
    y = root(y)
    if x == y:
        return
    par[x] += par[y]
    par[y] = x
 
def size(x):
    x = root(x)
    return -par[x]

N = int(input())
par = [-1] * N 
```
</details>

<details>
<summary><b>SortedSet</b></summary>

```python
import math
from bisect import bisect_left, bisect_right
from typing import Generic, Iterable, Iterator, TypeVar, Union, List
T = TypeVar('T')

class SortedSet(Generic[T]):
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def _build(self, a=None) -> None:
        "Evenly divide `a` into buckets."
        if a is None: a = list(self)
        size = self.size = len(a)
        bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
        self.a = [a[size * i // bucket_size : size * (i + 1) // bucket_size] for i in range(bucket_size)]
    
    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
        a = list(a)
        if not all(a[i] < a[i + 1] for i in range(len(a) - 1)):
            a = sorted(set(a))
        self._build(a)

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "SortedSet" + str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _find_bucket(self, x: T) -> List[T]:
        "Find the bucket which should contain x. self must not be empty."
        for a in self.a:
            if x <= a[-1]: return a
        return a

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        return i != len(a) and a[i] == x

    def add(self, x: T) -> bool:
        "Add an element and return True if added. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return True
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x: return False
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build()
        return True

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        if i == len(a) or a[i] != x: return False
        a.pop(i)
        self.size -= 1
        if len(a) == 0: self._build()
        return True
    
    def lt(self, x: T) -> Union[T, None]:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> Union[T, None]:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> Union[T, None]:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> Union[T, None]:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    
    def __getitem__(self, x: int) -> T:
        "Return the x-th element, or IndexError if it doesn't exist."
        if x < 0: x += self.size
        if x < 0: raise IndexError
        for a in self.a:
            if x < len(a): return a[x]
            x -= len(a)
        raise IndexError
    
    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans
```

> <b>s = SortedSet()</b> 初期化<br>
> <b>len(s)</b> sの長さを表示<br>
> <b>s[x]</b> 下からx番目の要素を返す<br>
> <b>s.add(n)</b> nを追加<br>
> <b>s.discard(n)</b> nを削除<br>
> <b>s.lt(n)</b> nより小さい最大の要素<br>
> <b>s.le(n)</b> n以下の最大の要素><br>
> <b>s.gt(n)</b> nより大きい最小の要素<br>
> <b>s.ge(n)</b> n以上の最小の要素<br>
> <b>s.index(n)</b> nより小さい要素の個数<br>
> <b>s.index_right(n)</b> n以下の要素の個数<br>

> (https://github.com/tatyam-prime/SortedSet)

</details>

<details>
<summary><b>SortedMultiset</b></summary>

```python
import math
from bisect import bisect_left, bisect_right, insort
from typing import Generic, Iterable, Iterator, TypeVar, Union, List
T = TypeVar('T')

class SortedMultiset(Generic[T]):
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def _build(self, a=None) -> None:
        "Evenly divide `a` into buckets."
        if a is None: a = list(self)
        size = self.size = len(a)
        bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
        self.a = [a[size * i // bucket_size : size * (i + 1) // bucket_size] for i in range(bucket_size)]
    
    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedMultiset from iterable. / O(N) if sorted / O(N log N)"
        a = list(a)
        if not all(a[i] <= a[i + 1] for i in range(len(a) - 1)):
            a = sorted(a)
        self._build(a)

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "SortedMultiset" + str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _find_bucket(self, x: T) -> List[T]:
        "Find the bucket which should contain x. self must not be empty."
        for a in self.a:
            if x <= a[-1]: return a
        return a

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        return i != len(a) and a[i] == x

    def count(self, x: T) -> int:
        "Count the number of x."
        return self.index_right(x) - self.index(x)

    def add(self, x: T) -> None:
        "Add an element. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return
        a = self._find_bucket(x)
        insort(a, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build()

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        if i == len(a) or a[i] != x: return False
        a.pop(i)
        self.size -= 1
        if len(a) == 0: self._build()
        return True

    def lt(self, x: T) -> Union[T, None]:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> Union[T, None]:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> Union[T, None]:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> Union[T, None]:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    
    def __getitem__(self, x: int) -> T:
        "Return the x-th element, or IndexError if it doesn't exist."
        if x < 0: x += self.size
        if x < 0: raise IndexError
        for a in self.a:
            if x < len(a): return a[x]
            x -= len(a)
        raise IndexError

    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans
```

> <b>s = SortedSet()</b> 初期化<br>
> <b>len(s)</b> sの長さを表示<br>
> <b>s[x]</b> 下からx番目の要素を返す<br>
> <b>s.add(n)</b> nを追加<br>
> <b>s.discard(n)</b> nを削除<br>
> <b>s.lt(n)</b> nより小さい最大の要素<br>
> <b>s.le(n)</b> n以下の最大の要素><br>
> <b>s.gt(n)</b> nより大きい最小の要素<br>
> <b>s.ge(n)</b> n以上の最小の要素<br>
> <b>s.index(n)</b> nより小さい要素の個数<br>
> <b>s.index_right(n)</b> n以下の要素の個数<br>
> <b>s.count(x)</b>sに含まれるxの個数を返す<br>

> (https://github.com/tatyam-prime/SortedSet)
</details>

<details>
<summary><b>強連結成分分解(SCC)</b></summary>

```python
import sys
sys.setrecursionlimit(10 ** 8)

def dfs(x):
    if come[x]: return
    come[x] = True
    for i in G[x]:
        dfs(i)
    backorder.append(x)

def rdfs(x):
    if come[x]: return
    come[x] = True
    components[-1].append(x)
    for i in rG[x]:
        rdfs(i)

N, M = map(int, input().split())
G = [[] for _ in range(N)]
rG = [[] for _ in range(N)]
for _ in range(M):
    a, b = map(lambda x: int(x) - 1, input().split())
    G[a].append(b)
    rG[b].append(a)

come = [False] * N
backorder = []

for i in range(N):
    if come[i]: continue
    dfs(i)

backorder.reverse()
come = [False] * N

# 強連結成分を格納するリスト
components = []

for i in backorder:
    if come[i]: continue
    components.append([])
    rdfs(i)
```

> [AtCoder 典型90 021](https://atcoder.jp/contests/typical90/tasks/typical90_u)<br>
> [Youtube かつっぱ競プロ](https://www.youtube.com/watch?v=cRbst-d4Fho&t=1198s)

</details>

<details>
<summary><b>凸包</b></summary>

```python
def cross_product(moto, saki0, saki1):
    # moto->saki0 の直線に対し saki1がどちら側にあるか
    # >0 ならば 左側 <0 ならば 右側
    x0 = saki0[0] - moto[0]
    y0 = saki0[1] - moto[1]
    x1 = saki1[0] - moto[0]
    y1 = saki1[1] - moto[1]
    cross_product = x0 * y1 - x1 * y0
    
    return cross_product

def wrap(ps):
    # ギフト包装法を使って凸包を求める。
    # 各点[x, y]をリストとして与えると凸包の各点をリストとして返す。
    qs = []
    # 最初の点
    x = [p[0] for p in ps]
    min_i = x.index(min(x))
    qs.append(ps[min_i]) # xが最小になる点をqs[0]とする。
    # 各点
    n = -1
    while True:
        n += 1
        for i in range(len(ps)):
            flag = False
            for p1 in ps:
                if qs[n] == ps[i]:
                    flag = True
                    break
                result = cross_product(qs[n], ps[i], p1)
                if result > 0 : # left
                    flag = True
                    break
            if flag == False:
                this_i = i
        if ps[this_i] == qs[0]:
            break
        qs.append(ps[this_i])
    
    return qs

# pointは座標をリスト型でまとめたもの
wrap(point)
```
</details>



## AtCoder Library

<details>
<summary><b>DSU(Disjoint Set Union)</b></summary>

> [Union Find]<br>
> 無向グラフで2頂点が連結かどうかを判定

### import
```python
from atcoder.dsu import DSU
```

### メソッド一覧
>uf = DSU(N) 初期化(Nは頂点数)<br>

>uf.merge(u, v) 頂点u,vの連結<br>
>uf.same(u, v) 頂点u,vの連結成分判定(True/False)<br>
>uf.leader(u) 頂点uのルート<br>
>uf.size(u) 頂点uの連結成分の頂点数<br>
>uf.groups() 各連結成分のリスト<br>

</details>

<details>
<summary><b>Fenwick Tree</b></summary>

>[Fenwick Tree(フェニック木)]<br>
>長さ $N$ のリスト $A$ に対して、<br>
>・リスト $A$ 内の要素 $A_i$ の値を変更する<br>
>・半開区間 $[l, r)$ の値の総和 $A_l+A_{l+1}+\cdots +A_{r-1}$ を求める<br>
> 上記の操作を$O(\log N)$で実行

### import
```python
from atcoder.fenwicktree import FenwickTree
```

### メソッド一覧
>ft = FenwickTree(N) 初期化(長さN、値0のリスト)<br>

>ft.add(i, v) Aiの値にvを加算<br>
>ft.sum(l, r) 半開区間[l,r)の値の総和を返す。
</details>

<details>
<summary><b>floor sum</b></summary>

> 次の式で表される値を、 $O(\log m)$ で求める<br>
> $\sum_{i = 0}^{n - 1} \left\lfloor \frac{a \times i + b}{m} \right\rfloor$<br>

### import
```python
from atcoder.math import floor_sum
```

### メソッド一覧
> floor_sum(n, m, a, b) 上記の式の値を返す<br>

</details>

<details>
<summary><b>SCC</b></summary>

> 強連結成分分解<br>
> 有向グラフにおいて、お互いに行き来できる頂点を1つのグループにまとめる<br>

### import
```python
from atcoder.scc import SCCGraph
```

### メソッド一覧
> graph = SCCGraph(N) 頂点数Nのグラフを作成<br>

> graph.add_edge(u, v) 頂点uから頂点vへの有向辺をはる<br>
> graph.scc() 各要素(リスト)は強連結成分を返す<br>

</details>

<details>
<summary><b>Segment Tree</b></summary>

> リスト内の区間に対する演算(総和、最大値、最小値など)の結果を返す $O(\log N)$ <br>

### import
```python
from atcoder.segtree import SegTree
```

### メソッド一覧
> st = SegTree(op, e, v) op:演算関数(sum, max, minなど), e:初期値, v: list型の場合はそのままのリスト、int型の場合はすべての要素がeで長さvのリスト<br>

> st.set(p, x)  リスト $A$ について、$A_p$ に $x$ を代入<br>
> st.get(p) リスト $A$ の $p$ 番目の要素 $A_p$<br>
> st.prod(l, r) 半開区間 $[l: r)$ における演算結果<br>
> st.all_prod() リスト全体における演算結果<br>
> st.max_right(p, func) セグメントツリー上で二分探索を行い、区間[x, j)がfuncを満たす最大のjを返す<br>
> st.min_left(p, func) セグメントツリー上で二分探索を行い、区間[j, p)がfuncを満たす最小のjを返す<br>

### 使用例
```python
from atcoder.segtree import SegTree

A = [1, 2, 3, 2, 1]
st = SegTree(max, -1, A)

print(st.min_left(0, lambda x: x < 3))
# 出力結果: 2 
#      → [0, 2)
```

</details>

[ac-library-python](https://github.com/not522/ac-library-python/blob/master/README_ja.md)<br>