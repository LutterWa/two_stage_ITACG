function parsave(i, p, restate)
para = num2cell(p);
[~, ~, ~, ~, ~, ~, qgt, qpt, ~, d] = deal(para{:});
filename = sprintf('data/flight_%d_qgt%d_qpt%d_d%d.mat', ...
    round(i), round(qgt), round(qpt), round(d));
save(filename, 'restate');
end
