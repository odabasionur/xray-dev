
1. Transfer modellere bak architecture çıkıyor mu çıkmıyorsa ..

2. xray interface olacak da arkada engine başka olacak mı ?
    1. protect edilecek attributeları belirle onlar için duruma göre getter setter oluştur

3. flatten gibi bir problemimiz var

4. Bütün module tiplerini tanımlayamayız. Dropout ayrı, pooling ayrı böyle olmaz. Generic bi şekilde yaz en kötü bi fonksiyon yaa.

5. Display yönteminin nasıl olacağını generic seçmek lazım. (node için mi filter için mi?)

6. Hook için ayrıca bir yerde bişeyler oluştur. Hepsi kendi içinde hook fonksiyonu oluşturmasın

7. istenilen layerların istenilen çıktıları alındı mı kontrol et. Alınmadıysa none de geç ama uyarı gönder

8. Selectorlara statik thresholdlar koyup, bu th'ların altında kaldığında tek frame'yerine birden fazla frame'i albüm 
gibi tek kare içinde birleştirerek paylaş

9. logger eklenecek mi

10. Warninglerde format string çalışmıyor

11. selectorlerde random seedler boşta

12. hepsine hook atıyoruz, sonra hepsinde tek tek selection yapıyoruz tek tek save ediyoruz. Hiç mantıklı değil ki.
Hiç mantıklı değil. Datalar büyükse nolcak. bir hook atıldığı anda işlemlerin yapılması sonra uçurulması lazımdı.

13. Save sistemini bi elden geçir ya bişey olmadı sanki. Gerçi neresi tam oldu ki

14. xraye kayıtlardan şunu aç diyince plotını bastırabilmeli

15. stabil bir model eğit

