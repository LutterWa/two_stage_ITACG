function output=rk4(state)
%ËÄ½×Áú¸ñ¿âËş·¨
h=0.02;
k1=h.*dery(state);
k2=h.*dery(state+0.5*k1);
k3=h.*dery(state+0.5*k2);
k4=h.*dery(state+k3);
output=state+(k1+2*k2+2*k3+k4)./6;
end
    
    
    