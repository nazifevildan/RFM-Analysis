
#Veriyi Anlama ve Hazırlama

import datetime as dt
import pandas as pd

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#flo_data_20K.csv verisini okutup,dataframe’in kopyasını oluşturduk.

df_ = pd.read_csv("dataset/flo_data_20k.csv")
df = df_.copy()

#Veri setinde;

#İlk 10 Gözlem

df.head(10)

#Değişken İsimleri

df.columns

# Betimsel istatistik

df.shape

#Boş değer

df.isnull().sum()  ##hiç boş değer yok

#Değişken tipleri

df.dtypes  ### 8 object, 4 float

#Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#alışveriş sayısı ve harcaması için yeni değişkenler oluşturduk.


df["master_id"].nunique()

df["total_num_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#Değişken tiplerini inceleyerek, tarih ifade eden değişkenlerin tipini date'e çevirdik.

df.dtypes
cols = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
df[cols] = df[cols].apply(pd.to_datetime)

#Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına baktık.

df.groupby("order_channel").agg({"master_id": "count",
                                 "total_num_omnichannel": "sum",
                                 "total_value_omnichannel": "sum"})

#En fazla kazancı getiren ilk 10 müşteriyi sıralattık.

df.groupby("master_id").agg({"total_value_omnichannel": "sum"}).sort_values("total_value_omnichannel",ascending=False).head(10)

df["total_value_omnichannel"].max()  # sağlama
df["total_value_omnichannel"].min()  # sağlama

#En fazla siparişi veren ilk 10 müşteriyi sıralattık.

df.groupby("master_id").agg({"total_num_omnichannel": "sum"}).sort_values("total_num_omnichannel", ascending=False).head(10)


# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.


def create_rfm(dataframe):
    dataframe["total_num_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe[
        "order_num_total_ever_offline"]
    dataframe["total_value_omnichannel"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    return dataframe


create_rfm(df)

#RFM Metriklerinin Hesaplanması

#Recency, Frequency ve Monetary tanımlarını;

""" Recency = Mevcut gün ile müşterinin son alım yaptığı gün arasında geçen zaman """
""" Frequency = Müşterinin toplam alım-sipariş sayısıdır """
""" Monetary = Müşterinin ilgili alışverişlerinden şirketin elde ettiği kazanç toplamı """

#recency
df["last_order_date"].dtypes
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

# frequency
df["total_num_omnichannel"]
# monetary
df["total_value_omnichannel"]

# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayıp,rfm isimli bir değişkene atadık.

rfm = df.groupby("master_id").agg({"last_order_date": lambda date: (today_date - date).dt.days,
                             "total_num_omnichannel": lambda num: num,
                             "total_value_omnichannel": lambda price: price})

#Oluşturduğumuz metriklerin isimlerini recency, frequency ve monetary olarak değiştirdik.

rfm.columns = ["recency", "frequency", "monetary"]

# monetarynin en düşük değerine baktım, eğer sıfırdan küçük değer olsaydı yok edecektik.
# rfm["monetary"].min()
# rfm = rfm[rfm["monetary"]>0]


#Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevirdik.
#Bu skorları recency_score, frequency_score ve monetary_score olarak kaydettik.

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#recency_score ve frequency_score’u tek bir değişken olarak ifade ederek ve RF_SCORE olarak kaydettik.

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# örn :

rfm[rfm["RF_SCORE"] == "55"]

#Oluşturulan RF skorları için segment tanımlamaları yaptık.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()

#Aksiyon Zamanı !

#Segmentlerin recency, frequnecy ve monetary ortalamalarını inceledik.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg("mean")

#RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu yüzden müşterilerin id numaralarını csv dosyasına kaydediyoruz.

rfm_2 = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["master_id"]

cust_ids = df[(df["master_id"].isin(rfm_2)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape


#Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediyoruz.

rfm_2_b  = rfm[rfm["segment"].isin(["cant_loose","about_to_sleep",
                                                          "new_customers"])]["master_id"]

cust_ids_b = df[(df["master_id"].isin(rfm_2_b)) &(df["interested_in_categories_12"].str.contains("ERKEK",case = False)) |
               (df["interested_in_categories_12"].str.contains("COCUK", case = False)) ]["master_id"]
cust_ids_b.to_csv("yeni_marka_hedef_müşteri_id_B.csv", index=False)