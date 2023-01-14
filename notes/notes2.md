

İyi kötü mimariyi çıkartmayı başardım.

Kalan kısım nasıl olmalı diye düşüne düşüne 2 günü yedim bitirdim. Ne olması gerektiğini bi kararlaştıralım.

1. **Xray class**\
Şimdi Xray genel bilgileri kullanıcından almalı ve interaction burada olmalı dedik.
Model, input, hangi layerda ne inspect edilecek falan bunları alacak veya değiştirecek falan

2. **Arch çıkarımı**\
model, inputu architechture tarafına verecek. Architecture çıkarma işlerini bu halledecek backprop veya forwarddan.

3. **Inspectin Belirlenmesi**\
Bu kısım net olmalı.
   1. Kullanıcı, architectureı bir dict olarak xray'den görebilmeli
   2. (bu ilk maddenin de bir açılımı aslında) Kullanıcı, hangi layerdan veya activationdan outputlar veya gradlar 
   save edilecek bunu xray'e söyleyebilmeli.
      1. Bunun bir yolu şu: Kullanıcı derki `MyModel`'da tanımlı şu şu attributeları track et diyebilir. 
      Ve bunların outputuna gradına weightine bak diyebilir. 
      2. conv layerların weightini al, activationların outputlarını falan olabilir belki bunu bilemiyorum.
      3. Diğer bir yolu: Kullanıcı, architecturı çağırdığı gibi default architecture'daki layerların tipine göre
      çıkarılmış bir draft da çıkarabilir. Şunun gibi 
      `{
      'conv2d-0': {'weight': True, 'output': False, 'grad': False}, 
      'relu-0': {'weight': False, 'output': True, 'grad': False},
      'Linear-1': {'weight': True, 'output': False, 'grad': False},
      }
      ` bişey. Sonrasında kullanıcı bunun üstünde True False ları değiştirebilir ve yeni halini set edebilir olmalı.
      Soru şu bunun için ve bu dictleri encapsulated olarak kendi içinde tutmak için yeni bir class olmalı mı. Yani Xray
      sadece interface olmuş olacak. Bu cümleyi okuyunca mantıklı geldi böyle olsun.

4. **Drafta göre çıtıların işlenmesi**\
   1. Hangi layera ne yapacağımızı aldık bir şekilde. Bundan sonra ne yapıcaz.
      1. output istenilenlere forward hook atılacak.
      2. weight isteilenler için model.named_parameters'a bakılmalı. Olum isimleri nerden bilcez lan. 
      Bunu da forward_hookla yapalım ya.
      3. grad için backward hook atılacak.

   2. Hook attık vs. elimizde bi ton layerın bi tonluk matrixleri oldu. Bunları direk alamayız. Bir takım selection 
   classları olsun. herbirinin make_selection() gibi bir methodu olsun. layerın atandığı classtan make_seleciton 
   çağırılsın. Böylece ileride kullanıcı kendi classını da yazabilir.\
   Layerın atanması şöyle olabilir. hangi layerdan hangi outputun default olarak çıktığını belirlediğimiz yerde bir de 
   bunu ekleriz. `{selection_method}: NaturalSelection` gibisine.
   
   3. Selection da yaptığımıza göre artık bunları save etmek kalıyor. Save kısmında önemli olan bir nevi metadata tutmak.
   modeli, hatta xray her çalıştığında unique bir id çıkartsın, Hangi layerdan, mümkünse hangi batchde, original_size, uzatılması gerekebilir bunun
   bunlarla birlikte veya bunlar ayrıca bir dosyada tutulmalı.


5. Son olarak bunu görselleştirmek lazım. Graph çekilir çekilmez bakılabilir olmalı. Kullanıcı istediği zaman 
tutup şu layerın şu çıktısını hatta şu batchdeki çıktısını getir diyebilmeli. Hatta hangi layerdan ne çıktı 
kaydetmiştik hangi batchlerde kayıt almıştık falan diyebilmeli

