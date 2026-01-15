var me = this;
var ME = $('#' + me.UUID)[0];

me.ready = function() {
  send_init(function(result){
    $(ME).append('<code>'+JSON.stringify(result)+'</code>');
  });
};
