clc;
clear;

M = 8;              %number of antennas at BS 
N = 32;             %number of elements at IRS
C = N+1;            %number of reflection patterns

j = sqrt(-1);
datalength_test = 20000;

Np = 20;                         %number of parfors
Nd = datalength_test / Np;      %number of examples per parfor

SNRdB = 5 ;
varNoise = 1; 

Nt = N+1;           %number of columns of H = [d B]

pl_f = 0.18^-2 / 31.6;
pl_G = 1/31.6;
pl_d = 1/33;

K_dB = 10 ;
K = 10^(K_dB/10) ;

Wn = exp ( j* 2*pi / C ) ; 
temp = 0:C-1;
s = [];

for tt = 0 : Nt - 1

    P_row = Wn.^(  tt * temp );

    s = [s; P_row];

end

parfor ii = 1 : Np
    
   tic

   data_SNR_test = [];
   label_SNR_test = [];
   
   count = 0;
    
while count < datalength_test / Np 

        count = count + 1 ;
                
        f = sqrt(pl_f) * 1/sqrt (2)* ( normrnd( 0,1,N,1 ) + j*normrnd( 0,1,N,1 ) );  % channel user to IRS
                
        G = sqrt(pl_G) * ( sqrt(K/(K+1)) + sqrt(1/(K+1)) * 1/sqrt (2)* ( normrnd( 0,1,M,N ) + j*normrnd( 0,1,M,N ) ) ) ;  %IRS to BS
                
        B = G * diag ( f );  %cascaded channel user to IRS to BS
        
        d = sqrt(pl_d) * 1/sqrt (2) * ( normrnd( 0,1,M,1 ) + j*normrnd( 0,1,M,1 ) );  % channel user to BS
        
        H =  [d B];  %total channel  
             
        Power = 10^( SNRdB / 10 ) * varNoise;  %transmission power
        
        S = sqrt( Power ) * s ;
        
        W =  sqrt( varNoise ) * 1/sqrt(2) * ( normrnd( 0, 1, M, C ) + j*normrnd( 0, 1, M, C ) ); 
        
        X = H*S + W ;   %received signal at BS 
        
        %LS Detection
        
        w_ls = S' / ( S * S' );
                       
        h_est_ls = X * w_ls;
        
        data_SNR_test( count, :, :, 1 ) = real ( h_est_ls ) ;
        data_SNR_test( count, :, :, 2 ) = imag ( h_est_ls ) ;
        label_SNR_test( count, :, :, 1 ) = real ( H ) ;  
        label_SNR_test( count, :, :, 2 ) = imag ( H ) ; 
                       
end

file_px = 'x_test_Rician_CSCG_';  
file_py = 'y_test_Rician_CSCG_';


file_x_test = strcat( file_px, num2str(ii),'_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );
file_h_test = strcat( file_py, num2str(ii),'_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );

m1=matfile( file_x_test,'writable',true );
m1.x_subtest =data_SNR_test;

m2=matfile( file_h_test,'writable',true );
m2.y_subtest = label_SNR_test;

toc
	
end

x_test = [];
y_test = [];
file_px = 'x_test_Rician_CSCG_';  
file_py = 'y_test_Rician_CSCG_';
file_px_temp = 'x_test_Rician_CSCG_K'; 
file_py_temp = 'y_test_Rician_CSCG_K';
file_px_res = strcat( file_px_temp, num2str(K_dB), 'dB_' );
file_py_res = strcat( file_py_temp, num2str(K_dB), 'dB_' );



for tt = 1 : Np
        
    temp1 = 1 + ( tt-1 ) * Nd;
    temp2 = Nd + ( tt-1 ) * Nd;
    
    temp = temp1 : temp2;
    
    file_x_test = strcat( file_px, num2str(tt), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat');  
    load( file_x_test );
    x_test( temp,:,:,: ) = x_subtest;    
    
    file_y_test = strcat( file_py, num2str(tt), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat');  
    load( file_y_test );
    y_test( temp,:,:,: ) = y_subtest;         

    delete( file_x_test );
    delete( file_y_test );
    
end


file_x_test_res = strcat( file_px_res, num2str( datalength_test ), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );   
save(file_x_test_res,'x_test');

file_y_test_res = strcat( file_py_res, num2str( datalength_test ), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );   
save(file_y_test_res,'y_test');

    
disp('testdata generated!')