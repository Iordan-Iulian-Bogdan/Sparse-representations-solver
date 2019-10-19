n=3840;
m=2160;

A=randn(m,n);
x=sprand(n,1,0.05); % generating a solution that has 5% non-zero elements
b=A*x;

file1=fopen('A.bin','w');
fwrite(file1,n,'int');
fwrite(file1,m,'int');
fwrite(file1,A','float');
fclose(file1);

file2=fopen('b.bin','w');
fwrite(file2,b,'float');
fclose(file2);

file3=fopen('sol.bin','w');
fwrite(file3,full(x),'float');
fclose(file3);
