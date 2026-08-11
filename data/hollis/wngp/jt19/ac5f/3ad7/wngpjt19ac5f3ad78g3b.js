var me = this;
var ME = $('#' + me.UUID)[0];

me.ready = function(api){
  send_cortex(function(result){
    $(ME).append('<PRE>'+JSON.stringify(result)+'</PRE>');
  });
};

/*
KEEP:

"You are the central monitor for a smart environment. You are receiving continuous audio, video, and sensor streams. Do not output anything if the environment is static. If a meaningful event occurs across any of your streams, output a real-time observation synthesizing what is happening."
*/