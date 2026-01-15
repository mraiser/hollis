var me = this;
var ME = $('#' + me.UUID)[0];

me.ready = function(api){
  send_cortex(function(result){
    $(ME).append('<PRE>'+JSON.stringify(result)+'</PRE>');
  });
};