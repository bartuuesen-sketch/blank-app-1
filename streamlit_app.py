import streamlit as st
import numpy as np
import pandas as pd

# --- Yardımcı Fonksiyon: String'den Sayıya Dönüştürme ---
def convert_to_float(value_str):
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    s = str(value_str).strip()
    if not s:
        raise ValueError("Boş değer.")
    
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0])
                denominator = float(parts[1])
                if denominator == 0:
                    raise ValueError("Bölen sıfır olamaz.")
                return numerator / denominator
            except ValueError:
                raise ValueError("Kesirdeki sayısal değerler geçersiz.")
        else:
            raise ValueError("Geçersiz kesir formatı. Örn: 1/3")
    else:
        try:
            return float(s)
        except ValueError:
            raise ValueError("Geçersiz sayısal format.")

# --- AHP Hesaplama Fonksiyonu ---
def calculate_ahp(pairwise_matrix, criterion_names):
    n = pairwise_matrix.shape[0]

    if n <= 1:
        return np.array([1.0]), 0.0, 0.0, 0.0, 0.0, True

    # Geometrik Ortalama
    geometric_means = np.prod(pairwise_matrix, axis=1)**(1/n)
    # Öncelik Vektörü
    priority_vector = geometric_means / np.sum(geometric_means)

    # Lambda Maksimum Hesaplaması
    weighted_sum_vector = np.dot(pairwise_matrix, priority_vector)
    
    lambda_max_values = []
    for ws, pv in zip(weighted_sum_vector, priority_vector):
        if pv != 0:
            lambda_max_values.append(ws / pv)
        else:
            lambda_max_values.append(0)
    
    lambda_max = np.mean(lambda_max_values)

    # Tutarlılık İndeksi (CI)
    consistency_index = (lambda_max - n) / (n - 1) if (n - 1) != 0 else 0

    # Rastgele Tutarlılık İndeksi (RI)
    random_index_dict = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    random_index = random_index_dict.get(n)

    # Tutarlılık Oranı (CR)
    if random_index is None:
        consistency_ratio = None
        is_consistent = False
    elif random_index == 0: 
        consistency_ratio = 0.0 if consistency_index == 0 else float('inf') 
        is_consistent = (consistency_index == 0)
    else:
        consistency_ratio = consistency_index / random_index
        is_consistent = consistency_ratio <= 0.10

    return priority_vector, lambda_max, consistency_index, random_index, consistency_ratio, is_consistent

# --- Tutarsızlık Analizi Fonksiyonu ---
def analyze_inconsistency(pairwise_matrix, criterion_names, priority_vector):
    """
    Tutarsızlığın hangi karşılaştırmalardan kaynaklandığını tespit eder
    """
    n = pairwise_matrix.shape[0]
    inconsistency_details = []
    
    # Her ikili karşılaştırma için tutarlılık kontrolü
    for i in range(n):
        for j in range(i+1, n):
            # Kullanıcının girdiği değer
            user_value = pairwise_matrix[i, j]
            
            # Teorik tutarlı değer (ağırlık oranı)
            if priority_vector[j] != 0:
                theoretical_value = priority_vector[i] / priority_vector[j]
            else:
                theoretical_value = user_value
            
            # Tutarsızlık derecesi hesapla
            if theoretical_value != 0:
                inconsistency_ratio = abs(user_value - theoretical_value) / theoretical_value
                
                # %20'den fazla sapma varsa problemli
                if inconsistency_ratio > 0.2:
                    inconsistency_details.append({
                        'kriter1': criterion_names[i],
                        'kriter2': criterion_names[j],
                        'girilen_deger': user_value,
                        'oneri_deger': theoretical_value,
                        'sapma_yuzdesi': inconsistency_ratio * 100,
                        'pozisyon': (i, j)
                    })
    
    # Sapmaya göre sırala (en problemli önce)
    inconsistency_details.sort(key=lambda x: x['sapma_yuzdesi'], reverse=True)
    
    return inconsistency_details

# --- Tutarlılık Önerileri Fonksiyonu ---
def suggest_consistency_improvements(inconsistency_details):
    """
    Tutarlılığı iyileştirmek için öneriler sunar
    """
    suggestions = []
    
    for detail in inconsistency_details[:5]:  # En problemli 5 tanesini göster
        kriter1 = detail['kriter1']
        kriter2 = detail['kriter2']
        girilen = detail['girilen_deger']
        oneri = detail['oneri_deger']
        sapma = detail['sapma_yuzdesi']
        
        if oneri > girilen:
            direction = "artırın"
            recommended_range = f"{oneri*0.8:.2f} - {oneri*1.2:.2f}"
        else:
            direction = "azaltın"
            recommended_range = f"{oneri*0.8:.2f} - {oneri*1.2:.2f}"
            
        suggestion = {
            'karsilastirma': f"{kriter1} vs {kriter2}",
            'sorun': f"Şu anki değer: {girilen:.2f}, Tutarlı değer: {oneri:.2f} civarında olmalı",
            'oneri': f"Bu değeri {direction} → Önerilen aralık: {recommended_range}",
            'sapma': f"%{sapma:.1f} sapma",
            'pozisyon': detail['pozisyon']
        }
        suggestions.append(suggestion)
    
    return suggestions

# --- TOPSIS Hesaplama Fonksiyonu ---
def calculate_topsis(decision_matrix_df, weights, criterion_types_dict):
    criterion_names = decision_matrix_df.columns.tolist()
    decision_matrix = decision_matrix_df.values.astype(float) 

    num_alternatives, num_criteria = decision_matrix.shape

    norm_factors = np.sqrt(np.sum(decision_matrix**2, axis=0))
    norm_factors[norm_factors == 0] = 1e-9 

    normalized_matrix = decision_matrix / norm_factors
    weighted_normalized_matrix = normalized_matrix * weights

    ideal_positive = np.zeros(num_criteria)
    ideal_negative = np.zeros(num_criteria)

    for j, criterion_name in enumerate(criterion_names):
        if criterion_types_dict.get(criterion_name) == 'Fayda':
            ideal_positive[j] = np.max(weighted_normalized_matrix[:, j])
            ideal_negative[j] = np.min(weighted_normalized_matrix[:, j])
        else: 
            ideal_positive[j] = np.min(weighted_normalized_matrix[:, j])
            ideal_negative[j] = np.max(weighted_normalized_matrix[:, j])

    distance_to_positive_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_positive)**2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_negative)**2, axis=1))

    denominator = (distance_to_positive_ideal + distance_to_negative_ideal)
    denominator[denominator == 0] = 1e-9 

    relative_closeness = distance_to_negative_ideal / denominator

    results_df = pd.DataFrame({
        'Tedarikçi': decision_matrix_df.index,
        'TOPSIS Skoru (Ci*)': relative_closeness
    })

    results_df = results_df.sort_values(by='TOPSIS Skoru (Ci*)', ascending=False).reset_index(drop=True)
    results_df['Sıra'] = results_df.index + 1

    return results_df

# --- Streamlit Uygulaması ---
st.set_page_config(
    layout="centered",
    page_title="Tedarikçi Değerlendirme Sistemi",
    page_icon="✅"
)

st.title("🤝 Tedarikçi Değerlendirme ve Seçim Sistemi")
st.markdown("""
Bu uygulama, Analitik Hiyerarşi Prosesi (AHP) ve TOPSIS metodolojilerini kullanarak
tedarikçi değerlendirme ve seçim süreçlerinizi optimize etmenize yardımcı olur.
""")

# Logo yoksa hata vermesin diye kontrol
try:
    st.image("mapa_logo.jpg", use_column_width=True, caption="Uygulama Logosu")
except:
    st.info("Logo dosyası bulunamadı, ancak uygulama normal çalışmaya devam ediyor.")

st.sidebar.header("Navigasyon")
st.sidebar.markdown("### AHP Kriter Ağırlıklandırma")
st.sidebar.markdown("### TOPSIS Tedarikçi Puanlama")

# AHP Bölümü
st.header("1. AHP Kriter Ağırlıklandırma")
st.write("Bu bölümde, AHP kullanarak tedarikçi seçim kriterlerinizin ağırlıklarını belirleyeceksiniz.")

num_criteria = st.number_input("Lütfen kriter sayısını girin:", min_value=2, max_value=10, value=4, step=1, key="num_criteria_input")

criterion_names = []
default_criterion_names = ["Maliyet", "Kalite", "Capability", "Flexibility", "Lojistik", "Sustainability", "Relations"]

col1, col2 = st.columns(2)
for i in range(int(num_criteria)):
    with col1 if i % 2 == 0 else col2:
        name = st.text_input(f"Kriter {i+1}:", key=f"kriter_isim_{i}", 
                             value=default_criterion_names[i] if i < len(default_criterion_names) else f"Kriter {i+1}")
        criterion_names.append(name)

st.write(f"**Tanımlanan Kriterler:** {', '.join(criterion_names)}")

st.header("2. İkili Karşılaştırma Matrisi")
st.markdown("""
**Önem Ölçeği:**
- 1: Eşit önemde  
- 3: Biraz daha önemli  
- 5: Oldukça önemli  
- 7: Çok önemli  
- 9: Aşırı derecede önemli  

**Nasıl doldurulur:** Sadece üst üçgendeki (beyaz) hücreleri doldurun. Alt üçgen otomatik hesaplanacak.
""")

# AHP matrisini başlat veya yeniden oluştur
if ('ahp_matrix' not in st.session_state or 
    st.session_state.get('last_num_criteria') != num_criteria):
    
    # Yeni matris oluştur
    matrix = np.ones((int(num_criteria), int(num_criteria)), dtype=float)
    st.session_state['ahp_matrix'] = matrix
    st.session_state['last_num_criteria'] = num_criteria

# Matris girişi için kolon düzeni
n = int(num_criteria)
matrix_cols = st.columns(n + 1)  # +1 kriter isimleri için

# Başlık satırı
with matrix_cols[0]:
    st.write("")  # Boş köşe
for j in range(n):
    with matrix_cols[j + 1]:
        st.write(f"**{criterion_names[j][:8]}**")  # İsmi kısalt

# Matris satırları
for i in range(n):
    matrix_cols = st.columns(n + 1)
    
    # Satır başlığı
    with matrix_cols[0]:
        st.write(f"**{criterion_names[i][:8]}**")
    
    for j in range(n):
        with matrix_cols[j + 1]:
            if i == j:
                # Köşegen - her zaman 1
                st.text_input("", value="1.0", disabled=True, 
                             key=f"matrix_{i}_{j}", label_visibility="collapsed")
            elif i < j:
                # Üst üçgen - kullanıcı girişi
                value = st.text_input("", value=str(st.session_state['ahp_matrix'][i, j]), 
                                     key=f"matrix_input_{i}_{j}", label_visibility="collapsed")
                try:
                    num_value = convert_to_float(value)
                    if num_value <= 0:
                        st.error("Pozitif sayı girin!")
                    else:
                        st.session_state['ahp_matrix'][i, j] = num_value
                        st.session_state['ahp_matrix'][j, i] = 1.0 / num_value
                except:
                    st.error("Geçersiz değer!")
            else:
                # Alt üçgen - otomatik hesaplanan
                reciprocal_value = st.session_state['ahp_matrix'][i, j]
                color = "background-color: #f0f0f0;"
                st.markdown(f'<div style="{color} padding: 0.25rem; border-radius: 0.25rem; text-align: center;">'
                           f'{reciprocal_value:.3f}</div>', unsafe_allow_html=True)

st.write("")  # Boşluk

# AHP Analizi
if st.button("🔍 AHP Analizini Başlat", type="primary"):
    try:
        priority_vector, lambda_max, ci, ri, cr, is_consistent = calculate_ahp(
            st.session_state['ahp_matrix'], criterion_names
        )
        
        # Sonuçları session state'e kaydet
        st.session_state['ahp_results'] = {
            'priority_vector': priority_vector,
            'lambda_max': lambda_max,
            'ci': ci,
            'ri': ri,
            'cr': cr,
            'is_consistent': is_consistent,
            'criterion_names': criterion_names.copy()
        }
        
        st.success("AHP analizi tamamlandı!")
        
    except Exception as e:
        st.error(f"AHP hesaplama hatası: {e}")

# AHP sonuçlarını göster
if 'ahp_results' in st.session_state:
    st.header("3. AHP Sonuçları")
    
    results = st.session_state['ahp_results']
    
    # Kriter ağırlıkları tablosu ve görselleştirme
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Kriter Ağırlıkları Tablosu")
        weights_df = pd.DataFrame({
            'Kriter': results['criterion_names'],
            'Ağırlık': [f"{w:.4f}" for w in results['priority_vector']],
            'Yüzde (%)': [f"{w*100:.2f}%" for w in results['priority_vector']]
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Ağırlık Dağılımı")
        # Pasta grafik için veri hazırlama
        chart_data = pd.DataFrame({
            'Kriter': results['criterion_names'],
            'Ağırlık': results['priority_vector']
        }).set_index('Kriter')
        st.bar_chart(chart_data)
    
    # Detaylı AHP matrisi sonuçları
    with st.expander("🔍 Detaylı AHP Analizi"):
        st.write("**Normalleştirilmiş İkili Karşılaştırma Matrisi:**")
        ahp_matrix_df = pd.DataFrame(
            st.session_state['ahp_matrix'], 
            columns=results['criterion_names'], 
            index=results['criterion_names']
        )
        st.dataframe(ahp_matrix_df.round(4), use_container_width=True)
    
    # Tutarlılık bilgileri ve iyileştirme önerileri
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lambda Max", f"{results['lambda_max']:.4f}")
    with col2:
        st.metric("Tutarlılık Oranı (CR)", f"{results['cr']:.4f}")
    with col3:
        if results['is_consistent']:
            st.success("✅ Tutarlı")
        else:
            st.error("❌ Tutarsız (CR > 0.10)")
    
    # Tutarsızlık analizi ve öneriler
    if not results['is_consistent']:
        st.warning("⚠️ **Tutarsızlık Tespit Edildi!**")
        
        # Tutarsızlık analizi yap
        inconsistency_details = analyze_inconsistency(
            st.session_state['ahp_matrix'], 
            results['criterion_names'], 
            results['priority_vector']
        )
        
        if inconsistency_details:
            # Önerileri göster
            suggestions = suggest_consistency_improvements(inconsistency_details)
            
            with st.expander("🔧 Tutarsızlık Düzeltme Önerileri", expanded=True):
                st.markdown("### 🎯 **En Problemli Karşılaştırmalar**")
                
                for i, suggestion in enumerate(suggestions, 1):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **{i}. {suggestion['karsilastirma']}**
                        - 🔍 **Sorun:** {suggestion['sorun']}
                        - 💡 **Öneri:** {suggestion['oneri']}
                        """)
                    
                    with col2:
                        st.error(suggestion['sapma'])
                
                st.markdown("""
                ### 📋 **Nasıl Düzeltirim?**
                1. **En yüksek sapmalı karşılaştırmaları** öncelikle düzeltin
                2. **Önerilen aralıktaki** değerleri deneyin  
                3. **AHP analizini tekrar çalıştırın**
                4. **CR < 0.10** olana kadar tekrarlayın
                
                ### 💡 **Tutarlılık İpuçları:**
                - **Geçişkenlik:** A>B ve B>C ise A>C olmalı
                - **Orantılılık:** A, B'den 3 kat önemli ve B, C'den 2 kat önemliyse A, C'den 6 kat önemli olmalı
                - **Aşırı değerlerden kaçının:** 9'dan büyük değerler tutarsızlığa yol açar
                """)
                
                # Kritik karşılaştırmaları vurgula
                if suggestions:
                    critical_positions = [s['pozisyon'] for s in suggestions[:3]]
                    st.info(f"🎯 **Öncelikle düzeltilmesi gerekenler:** " + 
                           ", ".join([f"{results['criterion_names'][pos[0]]} vs {results['criterion_names'][pos[1]]}" 
                                    for pos in critical_positions]))
        else:
            st.info("Tutarsızlık tespit edildi ancak spesifik problem noktaları bulunamadı. Genel olarak değerleri gözden geçirin.")
    
    # TOPSIS Bölümü
    if results['is_consistent']:
        st.header("4. TOPSIS Tedarikçi Değerlendirme")
        
        # Tedarikçi sayısı
        num_suppliers = st.number_input("Tedarikçi sayısı:", min_value=2, max_value=20, value=3)
        
        # Tedarikçi isimleri
        supplier_names = []
        supplier_cols = st.columns(min(3, int(num_suppliers)))
        for i in range(int(num_suppliers)):
            with supplier_cols[i % 3]:
                name = st.text_input(f"Tedarikçi {i+1}:", value=f"Tedarikçi_{i+1}", key=f"supplier_{i}")
                supplier_names.append(name)
        
        # Kriter tipleri açıklaması
        st.subheader("Kriter Tipleri")
        
        with st.expander("📖 Fayda vs Maliyet Kriterleri Nedir?", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🟢 FAYDA KRİTERLERİ**
                - **Yüksek değer = İyi**
                - Maksimize edilmek istenir
                - **Örnekler:**
                  - Kalite puanı
                  - Müşteri memnuniyeti
                  - Teslimat hızı
                  - Esneklik
                  - Sürdürülebilirlik puanı
                """)
            with col2:
                st.markdown("""
                **🔴 MALİYET KRİTERLERİ**
                - **Düşük değer = İyi**
                - Minimize edilmek istenir
                - **Örnekler:**
                  - Fiyat/Maliyet
                  - Teslimat süresi (gün)
                  - Hata oranı (%)
                  - Risk seviyesi
                  - Mesafe (km)
                """)
        
        # Kriter tipi seçimi
        criterion_types = {}
        st.write("Her kriter için uygun tipi seçin:")
        type_cols = st.columns(2)
        for i, criterion in enumerate(results['criterion_names']):
            with type_cols[i % 2]:
                # Varsayılan tip önerisi
                default_type = "Maliyet" if "maliyet" in criterion.lower() or "fiyat" in criterion.lower() else "Fayda"
                criterion_types[criterion] = st.selectbox(
                    f"**{criterion}**:", 
                    ["Fayda", "Maliyet"],
                    index=0 if default_type == "Fayda" else 1,
                    key=f"type_{criterion}",
                    help=f"Bu kriter için yüksek değer iyi mi (Fayda) yoksa düşük değer iyi mi (Maliyet)?"
                )
        
        # Kriter değerlendirme tiplerini seç
        st.subheader("Değerlendirme Tipleri")
        st.write("Her kriter için nasıl değerlendirme yapacağınızı seçin:")
        
        evaluation_types = {}
        eval_cols = st.columns(2)
        for i, criterion in enumerate(results['criterion_names']):
            with eval_cols[i % 2]:
                eval_type = st.selectbox(
                    f"**{criterion}** değerlendirme tipi:",
                    ["Sayısal Değer", "1-10 Skala"],
                    key=f"eval_type_{criterion}",
                    help="Gerçek sayısal veri mi gireceksiniz yoksa 1-10 arası subjektif puan mı?"
                )
                evaluation_types[criterion] = eval_type
        
        # Performans matrisi girişi
        st.subheader("Tedarikçi Performans Değerleri")
        
        with st.expander("📝 Değerlendirme Rehberi - Nasıl Veri Gireceğim?", expanded=True):
            st.markdown("""
            ### 📊 **İki Tip Değerlendirme Modu**
            
            #### 🔢 **SAYISAL DEĞER Modu**
            **Gerçek verileri doğrudan girin:**
            - **Fiyat:** 15000, 18500, 22000 (TL cinsinden)
            - **Teslimat Süresi:** 3, 5, 7 (gün cinsinden)  
            - **Kapasite:** 1000, 1500, 800 (adet/ay)
            - **Hata Oranı:** 0.5, 1.2, 0.8 (% cinsinden)
            - **Mesafe:** 50, 150, 300 (km cinsinden)
            
            #### 🎯 **1-10 SKALA Modu**  
            **Subjektif değerlendirmeler için:**
            
            | Puan | Seviye | Ne Anlama Gelir? |
            |------|--------|------------------|
            | **9-10** | 🌟 **Mükemmel** | Sektör lideri, örnek performans |
            | **7-8** | ✅ **İyi** | Beklentileri karşılıyor, güvenilir |
            | **5-6** | ⚡ **Orta** | Kabul edilebilir, ortalama |
            | **3-4** | ⚠️ **Zayıf** | Beklentilerin altında |
            | **1-2** | ❌ **Çok Zayıf** | Kabul edilemez |
            
            ### 💡 **Önemli Notlar:**
            - **TOPSIS normalizasyon yapar** → Farklı birimler (TL, gün, %) birlikte kullanılabilir
            - **Fayda kriterleri:** Yüksek değer iyi
            - **Maliyet kriterleri:** Düşük değer iyi (örn: düşük fiyat = iyi)
            - **Karşılaştırmalı düşünün:** Tedarikçiler arasındaki farklar önemli
            """)
        
        st.write("**Her tedarikçi için performans verilerini girin:**")
        
        # Performans matrisini başlat
        if ('performance_matrix' not in st.session_state or 
            len(supplier_names) != st.session_state.get('last_num_suppliers', 0)):
            
            performance_matrix = np.ones((len(supplier_names), len(results['criterion_names'])), dtype=float) * 5
            st.session_state['performance_matrix'] = performance_matrix
            st.session_state['last_num_suppliers'] = len(supplier_names)
        
        # Performans matrisi girişi - Dinamik input tipleri
        perf_data = []
        for i, supplier in enumerate(supplier_names):
            row = []
            cols = st.columns(len(results['criterion_names']) + 1)
            
            with cols[0]:
                st.write(f"**{supplier}**")
            
            for j, criterion in enumerate(results['criterion_names']):
                with cols[j + 1]:
                    if evaluation_types[criterion] == "Sayısal Değer":
                        # Sayısal değer girişi - birim önerisi
                        unit_suggestions = {
                            "maliyet": "TL",
                            "fiyat": "TL", 
                            "süre": "gün",
                            "zaman": "gün",
                            "mesafe": "km",
                            "kapasite": "adet",
                            "oran": "%",
                            "hız": "km/h"
                        }
                        
                        # Birim önerisi
                        suggested_unit = ""
                        for keyword, unit in unit_suggestions.items():
                            if keyword in criterion.lower():
                                suggested_unit = f" ({unit})"
                                break
                        
                        value = st.number_input(
                            f"{criterion[:8]}{suggested_unit}", 
                            min_value=0.0,
                            value=float(st.session_state['performance_matrix'][i, j]) if evaluation_types[criterion] == "Sayısal Değer" else 5.0,
                            step=0.01,
                            format="%.2f",
                            key=f"perf_{i}_{j}",
                            label_visibility="visible",
                            help=f"Bu kriter için {supplier}'in gerçek değerini girin"
                        )
                    else:
                        # 1-10 skala girişi
                        value = st.number_input(
                            f"{criterion[:8]} (1-10)", 
                            min_value=1.0, 
                            max_value=10.0, 
                            value=float(st.session_state['performance_matrix'][i, j]) if evaluation_types[criterion] == "1-10 Skala" else 5.0,
                            step=0.1,
                            key=f"perf_{i}_{j}",
                            label_visibility="visible",
                            help=f"1=Çok Kötü, 10=Mükemmel"
                        )
                    
                    row.append(value)
                    st.session_state['performance_matrix'][i, j] = value
            perf_data.append(row)
        
        # TOPSIS analizi
        if st.button("🎯 TOPSIS Analizini Başlat", type="primary"):
            try:
                # Performans matrisi DataFrame'e dönüştür
                performance_df = pd.DataFrame(
                    st.session_state['performance_matrix'],
                    index=supplier_names,
                    columns=results['criterion_names']
                )
                
                # TOPSIS hesapla
                topsis_results = calculate_topsis(
                    performance_df, 
                    results['priority_vector'], 
                    criterion_types
                )
                
                st.header("5. 🏆 TOPSIS Sonuçları")
                
                # Sonuç tablosu - daha güzel görünüm
                st.subheader("📋 Tedarikçi Sıralaması")
                
                # Sonuç tablosunu güzelleştir
                display_results = topsis_results.copy()
                display_results['TOPSIS Skoru (Ci*)'] = display_results['TOPSIS Skoru (Ci*)'].round(4)
                display_results['Skor (%)'] = (display_results['TOPSIS Skoru (Ci*)'] * 100).round(2)
                
                # Renk kodlaması için
                def highlight_best(row):
                    if row['Sıra'] == 1:
                        return ['background-color: #90EE90'] * len(row)  # Açık yeşil
                    elif row['Sıra'] == 2:
                        return ['background-color: #FFE4B5'] * len(row)  # Açık turuncu
                    elif row['Sıra'] == 3:
                        return ['background-color: #FFB6C1'] * len(row)  # Açık pembe
                    else:
                        return [''] * len(row)
                
                styled_results = display_results.style.apply(highlight_best, axis=1)
                st.dataframe(styled_results, use_container_width=True, hide_index=True)
                
                # En iyi tedarikçiler
                col1, col2, col3 = st.columns(3)
                with col1:
                    best = topsis_results.iloc[0]
                    st.metric(
                        "🥇 1. Sıra", 
                        best['Tedarikçi'],
                        f"Skor: {best['TOPSIS Skoru (Ci*)']:.4f}"
                    )
                
                if len(topsis_results) > 1:
                    with col2:
                        second = topsis_results.iloc[1] 
                        st.metric(
                            "🥈 2. Sıra", 
                            second['Tedarikçi'],
                            f"Skor: {second['TOPSIS Skoru (Ci*)']:.4f}"
                        )
                
                if len(topsis_results) > 2:
                    with col3:
                        third = topsis_results.iloc[2]
                        st.metric(
                            "🥉 3. Sıra", 
                            third['Tedarikçi'],
                            f"Skor: {third['TOPSIS Skoru (Ci*)']:.4f}"
                        )
                
                # Görselleştirme
                st.subheader("📊 TOPSIS Skor Dağılımı")
                chart_data = topsis_results.set_index('Tedarikçi')['TOPSIS Skoru (Ci*)']
                st.bar_chart(chart_data)
                
                # Detaylı TOPSIS analizi
                with st.expander("🔍 Detaylı TOPSIS Analizi"):
                    st.write("**📊 Girilen Ham Performans Verileri:**")
                    perf_display = pd.DataFrame(
                        st.session_state['performance_matrix'],
                        index=supplier_names,
                        columns=results['criterion_names']
                    )
                    
                    # Değerlendirme tiplerine göre formatla
                    formatted_perf = perf_display.copy()
                    for criterion in results['criterion_names']:
                        if evaluation_types[criterion] == "Sayısal Değer":
                            formatted_perf[criterion] = formatted_perf[criterion].apply(lambda x: f"{x:.2f}")
                        else:
                            formatted_perf[criterion] = formatted_perf[criterion].apply(lambda x: f"{x:.1f}/10")
                    
                    st.dataframe(formatted_perf, use_container_width=True)
                    
                    st.write("**⚙️ TOPSIS Matematiksel İşlem Süreci:**")
                    st.markdown("""
                    1. **Normalizasyon:** Her kriter için √(Σx²) ile bölerek [0,1] aralığına getirme
                    2. **Ağırlıklandırma:** AHP ağırlıkları ile çarpma  
                    3. **İdeal Çözümler:**
                       - **A+:** Fayda kriterleri için MAX, Maliyet kriterleri için MIN
                       - **A-:** Fayda kriterleri için MIN, Maliyet kriterleri için MAX
                    4. **Öklid Uzaklıkları:** Her tedarikçinin A+ ve A-'ye uzaklığı
                    5. **Göreceli Yakınlık:** Ci* = Si⁻/(Si⁺ + Si⁻) → [0,1] arası skor
                    """)
                    
                    st.write("**📋 Kriter Tipleri, Değerlendirme Modları ve Ağırlıkları:**")
                    criteria_info = pd.DataFrame({
                        'Kriter': results['criterion_names'],
                        'Tip': [criterion_types[c] for c in results['criterion_names']],
                        'Değerlendirme': [evaluation_types[c] for c in results['criterion_names']],
                        'Ağırlık': [f"{w:.4f}" for w in results['priority_vector']],
                        'Ağırlık (%)': [f"{w*100:.2f}%" for w in results['priority_vector']]
                    })
                    st.dataframe(criteria_info, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"TOPSIS hesaplama hatası: {e}")
    else:
        st.warning("⚠️ TOPSIS analizi için önce tutarlı bir AHP matrisi gereklidir.")

# Bilgilendirme
with st.expander("ℹ️ Nasıl Kullanılır?"):
    st.markdown("""
    1. **Kriter Tanımlama:** İstediğiniz kriterleri tanımlayın
    2. **İkili Karşılaştırma:** Üst üçgendeki hücrelere değerler girin
    3. **AHP Analizi:** Kriter ağırlıklarını hesaplayın
    4. **TOPSIS Analizi:** Tedarikçileri değerlendirin ve sıralayın
    
    **İpucu:** Kesirli değerler için `1/3`, `0.33` gibi formatları kullanabilirsiniz.
    """)
