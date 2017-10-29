function smprime = SoftmaxPrime(act_value,target)
  [M,I]=max(target);
  smprime=act_value(I);
end
