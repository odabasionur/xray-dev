# Hikaye

Amaç şu, kullanıcı training aşamalarını yazdıktan sonra, nasıl ki validation ölçümleri için ayrı bi for açıyor, 
orada mümkün olduğunca az eforla, mümkünse tek satırla grafikleri çıkart diyebilmeli.

```
layers_to_inspect: list[str] = ['conv1d', 'conv3d', 'fc2']  # which must be defined in
                                                            # model(custom nn.module class) as attribute
dict_of_layers_to_inspect: dict[dict[str: bool]] = {
    'conv1d': {'weight': True, 'layer': True, 'activation': True, 'grad': True},
    'conv3d': {'weight': True, 'layer': False, 'activation': True, 'grad': False},
    'fc1': {'weight': True, 'layer': False, 'activation': True, 'grad': False}
    }

xray = Xray(model, (input_size or input), (layers_to_inspect, dict_of_layers_to_inspect, None))

for epoch_num in range(10):
    for batch_num, (x, y) in enumerate(dataloader):
        x.to(device)
        y.to(device)
        
        optim.zero_grad()
        
        loss = model(x)
        loss.backward()
        
        if batch_num % 100 == 0:
            # some validation metric calculations like 
            # loss_val = model(x_val) and so on ..
            
            xray.take_graphs()
            xray.show_last_graphs()     # take_graph'ın parametresi bile olabilir 
```

Yani kodda hangi layerdan ne bilgi çıkarılacağı yazıldıktan sonra toplam **2 satır gerekmeli**.


## Kim ne yapıyor

Xray toparlayıcı olacak. Kullanıcı sadece burayı kullanarak işini görebilmeli. Bazı fonksiyonları ayrıca kullanılabilir.
Xray modeli ve input_size ve inputu alır. Bunları, forward hookları kullanarak mimari yapıyı çıkartmada kullanır. 
Ayrıca, modelden çıkan outputla birlikte grad_fnleri takip ederek backward yolunu çıkartabilir. 

### Architecutre Paradoksu
* Grad_fn' in outputtan inputa kadar yolu eksiksiz çıkartılabiliyor. 
* Ama grad fn'in hangi layera bağlı olduğu bilinmiyor. İsminden bir conv layera ait olduğu varsayılabilir ama hangi convlayer?
* Layer'dan da grad_fn'e doğrudan ilişki kurulamıyor ama layerın outputunun grad_fn'i biliniyor.
Diğer taraftan ise, forwardda girilecek tüm layerlar bilinmiyor. 
Hani, ben tüm layerların outputundan grad_fn'i çıkartırım ki diyemiyosun.\

Can alıcı soru: Peki yapan nasıl yapıyor?\
Valla ben pek anlamadım. Kodlar çok karışık geldi. Kendim daha çabuk yaparım dedim, yapamadım.

Bu paradoksun bir şekilde çözüldüğünü varsayarak;

* ArchitectureTracer (gibisine) bir class yukarıdaki paradoksu çözecek ve architecture'ı kimin neye bağlı olduğunu bulacak.
Ayrıca, sahip olduğu bir class'ta hook atma methodları olacak. 
Xray'e hangi layerların alınacağı söylenmezse, ArchitectureTracer'a başvuracak ve default olarak conv layerların outputunu
ve activation'ını, fc layerların sadece activation'ını alacak şekilde kendisi bir `dict_of_layers_to_inspect` çıkaracak.

* FileManager: path, dosya oluşturma, matrixleri efektif bir şekilde diske yazma/okuma işlerini halledecek. Ayrıca,
hangi layerın kaçıncı timestmapindeki ne dosyası nerede tutuluyor bunu takip edecek ve edilebilecek şekilde yazaca artık.


Buraya kadarki kısım initte olacaklar. xray.take_graph() diyince ne olmalı:
* layer output ve activation output için (bunlar ayrı ayrı mı tutulmalı ya cidden? müşterisi çıkar mı yani?):
  * forwarda hook bağlanmalı **Muhtemelen hook atıldıktan sonraki model(input) ile çıktılar alınacak**
  * sonuçlar bir şekilde bir dict'e vs. yazılacak. Fakat bu çok büyük bir matrix olabilir. Onun için burada
  bazı statik thresholdlar olmalı. Bu thresholdların altındaysa alayını işleme al. Değilse bir class'tan yardım almalı
  ki defaultta `RandomFilterSelectorType` olacaktır (değişebilir). Ve sonuç olarak kayıt edilebilecek sayıda bir takım
  Tensorler bulunacak.
  * Bu Tensorler filemanagere verilip makul bir şekilde save edilecek.
  * Olurda kullanıcı sabırsız çıkar hemen bi bakıyım derse hazır ram'deyken gösteriverelim.
  
* weight'ın bizzat kendisi:
  * hook mook gerektirmez. model.named_parameters() için çek getir deriz.
  * ama büyük matrix problemleri yine çıkacak. Fc de olsa conv da olsa. Fc için ayrı Conv için ayrı threshold ve
  selector class olmalı veya `RandomFilterSelectorType` içinde olabilir veya adı değiştirilir bu selector classların.
    * 

* gradient output:
  * burada bi backward hook atmak gerekecek. Onu biliyorum.
  * Sanıyorum gerisi yine outputu aldın threshold çaktın baktın olmuyo Selector gillere ver sonra kaydet olur yani



Hadi diyelim bunları da yaptık. Varsaymazsak ilerleyemiyoruz zira. 
Birde artık en son adam diyecek ki hadi göster ne kaydettin
Sadece activation output almış olsak ve 2 tane 4 tane gibi küçük sayılar almış olsak. 10 tane de layer olsa. 
100 kere de take_graph demiş olsa. bunu nasıl göstercez. Neyi göstercez yani. Bu soruyu şu an soruyor olmam daha bi tuhaf.
Hadi bi gif yapacaz dedik bu tamam da. gif yoksa göstermiycez mi? Yada bi klasöre save edelim git bak mı diyelim. 
* Bari şöyle diyelim, adam layer adını falan versin timestampi versin
(toplam kaç timestamp ve hangi layerdan neleri (weighs mi output mu falan) ne kadar save ettiğimizi dönmemiz lazım), bizde onu show edelim.









# **VARSAYIMLAR**
* inspect edilecek bütün layerların bir nn.Module olması gerekiyor. diğer türlü hook atamıyoruz. Örneğin flatten.\
(flattenın normalden farklı olarak `._register_...()` methodları vardı. `model.apply()` yerin `named_modules()` 
ile deneyip bu istisna düzeltilebilir mi ?)

* inspect edilecek bütün layerların initte tanımlanması gerekiyor. `self.relu = nn.ReLU()` şeklinde. Tekrar kullanımlar takip edilebiliyor.

* sequential modellere hiç bakmadım (`nn.Sequential`). Yer mi uyarlamak gerekecek mi bilmiyorum


