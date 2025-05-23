function para = latin(n, d, lob, upb)
rng(1)  % 设置随机种子
for i = 1 : d % for each dimensions
    A(:, i) = randperm(n)'; % get random value range for each dimensions
end
t = (A - rand(n, d)) / n; % get position for every value range
span = upb - lob;
z1 = zeros(n, d);
z2 = zeros(n, d);
for i = 1 : n
    z1(i, :) = span;
    z2(i, :) = lob;
end
para = t .* z1 + z2;
end