function z = condEntropy (x, y)

assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); 
Hxy = -dot(Pxy,log(Pxy));

Py = nonzeros(mean(My,1));
Hy = -dot(Py,log(Py));

% conditional entropy H(x|y)
z = Hxy-Hy;
z = max(0,z);
