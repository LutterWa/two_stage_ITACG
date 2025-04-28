function output=rk4(state, atm, Ma2)
%ËÄ½×Áú¸ñ¿âËş·¨
h=0.02;
k1=h.*dery(state, atm, Ma2);
k2=h.*dery(state+0.5*k1, atm, Ma2);
k3=h.*dery(state+0.5*k2, atm, Ma2);
k4=h.*dery(state+k3, atm, Ma2);
output=state+(k1+2*k2+2*k3+k4)./6;
end
    
    
    