"""
Temp klasöründeki eski dosyaları temizle
Sadece en son checkpoint ve examples dosyalarını tutar

⚠️ UYARI: Program çalışırken çalıştırmayın! Sadece program durduğunda kullanın.
"""
import os
import glob
import time
from pathlib import Path

# Program çalışırken KULLANILMAMASI GEREKEN kritik dosyalar
CRITICAL_FILES = [
    'temp.pth.tar',
    'best.pth.tar',
    'latest.pth.tar',
    'best.pth.tar.examples',
    'latest.pth.tar.examples',
]

def is_file_locked(filepath):
    """Dosyanın kilitli olup olmadığını kontrol et (basit kontrol)"""
    try:
        # Dosyayı açmayı dene (read mode)
        with open(filepath, 'r+b'):
            return False
    except (IOError, OSError, PermissionError):
        return True

def cleanup_temp_folder(temp_folder='./temp/', keep_checkpoints=1, keep_examples=2, skip_critical=True):
    """
    Temp klasöründeki eski dosyaları temizle
    
    Args:
        temp_folder: Temp klasör yolu
        keep_checkpoints: Tutulacak checkpoint sayısı
        keep_examples: Tutulacak examples dosyası sayısı
        skip_critical: Kritik dosyaları atla (True ise temp.pth.tar, best.pth.tar, latest.pth.tar silinmez)
    """
    if not os.path.exists(temp_folder):
        print(f"Klasör bulunamadı: {temp_folder}")
        return
    
    if skip_critical:
        print("⚠️  Kritik dosyalar korunuyor (temp.pth.tar, best.pth.tar, latest.pth.tar)")
    
    total_size_before = sum(f.stat().st_size for f in Path(temp_folder).rglob('*') if f.is_file())
    
    # Checkpoint dosyalarını temizle
    checkpoint_pattern = os.path.join(temp_folder, 'checkpoint_*.pth.tar')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    def get_iteration_num(filename):
        try:
            basename = os.path.basename(filename)
            num_str = basename.replace('checkpoint_', '').replace('.pth.tar', '')
            return int(num_str)
        except:
            return -1
    
    checkpoint_files.sort(key=get_iteration_num, reverse=True)
    
    # Son N checkpoint'i tut, gerisini sil
    if len(checkpoint_files) > keep_checkpoints:
        files_to_delete = checkpoint_files[keep_checkpoints:]
        for f in files_to_delete:
            try:
                filename = os.path.basename(f)
                
                # Kritik dosyaları atla
                if skip_critical and filename in CRITICAL_FILES:
                    print(f'⊘ Atlanıyor (kritik): {filename}')
                    continue
                
                # Dosya kilitli mi kontrol et
                if is_file_locked(f):
                    print(f'⚠️  Atlanıyor (kilitli): {filename} - Program kullanıyor olabilir!')
                    continue
                
                size = os.path.getsize(f)
                os.remove(f)
                print(f'✓ Silindi: {filename} ({size/1024/1024:.2f} MB)')
                
                # İlgili examples dosyasını da sil
                examples_file = f + '.examples'
                if os.path.exists(examples_file):
                    examples_filename = os.path.basename(examples_file)
                    if skip_critical and examples_filename in CRITICAL_FILES:
                        print(f'⊘ Atlanıyor (kritik): {examples_filename}')
                    elif is_file_locked(examples_file):
                        print(f'⚠️  Atlanıyor (kilitli): {examples_filename}')
                    else:
                        size = os.path.getsize(examples_file)
                        os.remove(examples_file)
                        print(f'✓ Silindi: {examples_filename} ({size/1024/1024:.2f} MB)')
            except PermissionError as e:
                print(f'⚠️  İzin hatası (dosya kullanılıyor olabilir): {os.path.basename(f)}')
            except Exception as e:
                print(f'✗ Silinemedi: {os.path.basename(f)} - {e}')
    
    # Iteration examples dosyalarını temizle
    iteration_pattern = os.path.join(temp_folder, 'iteration_*.examples')
    iteration_files = glob.glob(iteration_pattern)
    
    def get_iteration_num_examples(filename):
        try:
            basename = os.path.basename(filename)
            num_str = basename.replace('iteration_', '').replace('.examples', '')
            return int(num_str)
        except:
            return -1
    
    iteration_files.sort(key=get_iteration_num_examples, reverse=True)
    
    if len(iteration_files) > keep_examples:
        files_to_delete = iteration_files[keep_examples:]
        for f in files_to_delete:
            try:
                filename = os.path.basename(f)
                
                # Kritik dosyaları atla
                if skip_critical and filename in CRITICAL_FILES:
                    print(f'⊘ Atlanıyor (kritik): {filename}')
                    continue
                
                # Dosya kilitli mi kontrol et
                if is_file_locked(f):
                    print(f'⚠️  Atlanıyor (kilitli): {filename} - Program kullanıyor olabilir!')
                    continue
                
                size = os.path.getsize(f)
                os.remove(f)
                print(f'✓ Silindi: {filename} ({size/1024/1024:.2f} MB)')
            except PermissionError as e:
                print(f'⚠️  İzin hatası (dosya kullanılıyor olabilir): {os.path.basename(f)}')
            except Exception as e:
                print(f'✗ Silinemedi: {os.path.basename(f)} - {e}')
    
    total_size_after = sum(f.stat().st_size for f in Path(temp_folder).rglob('*') if f.is_file())
    freed_space = total_size_before - total_size_after
    
    print(f"\n📊 Özet:")
    print(f"   Önceki boyut: {total_size_before/1024/1024:.2f} MB")
    print(f"   Sonraki boyut: {total_size_after/1024/1024:.2f} MB")
    print(f"   Temizlenen: {freed_space/1024/1024:.2f} MB")

if __name__ == "__main__":
    print("🧹 Temp klasörü temizleniyor...")
    print("=" * 50)
    print("⚠️  UYARI: Program çalışırken çalıştırmayın!")
    print("    Kritik dosyalar (temp.pth.tar, best.pth.tar, latest.pth.tar) korunacak.")
    print("=" * 50)
    time.sleep(2)  # Kullanıcıya okuma fırsatı ver
    
    # Yer sıkıntısı için agresif temizlik: sadece kritik dosyaları tut
    cleanup_temp_folder(keep_checkpoints=0, keep_examples=0, skip_critical=True)
    print("=" * 50)
    print("✅ Temizleme tamamlandı!")

