import os
import sys
import warnings
from faster_whisper import WhisperModel

# python whisper_demo.py data/audio/1月20日_1415_会议-5d75ac50.mp3
# 忽略无关警告
warnings.filterwarnings("ignore")

# ========== 在线模式核心配置（确保能下载模型） ==========
os.environ["HF_HUB_OFFLINE"] = "0"          # 强制在线
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 清华镜像
# 如需代理，取消下面注释并替换为你的代理地址
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

class OnlineWhisperFix:
    """
    能正常运行的纯在线模式 Whisper 识别器（修复段错误）
    """
    def __init__(self, model_size="tiny"):
        # 关键1：禁用int8，改用float32（兼容所有CPU）
        self.compute_type = "float32"
        self.model_alias = {
            "tiny": "Systran/faster-whisper-tiny",
            "base": "Systran/faster-whisper-base"
        }
        self.model_name = self.model_alias[model_size.lower()]

        try:
            print(f"📥 在线加载 {model_size} 模型（compute_type={self.compute_type}）...")
            # 关键2：强制单线程，避免多线程内存冲突
            self.model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
                local_files_only=False,  # 纯在线模式
                cpu_threads=1  # 禁用多线程，解决段错误核心
            )
            print(f"✅ {model_size} 模型在线加载成功！")
        except Exception as e:
            raise RuntimeError(f"❌ 在线加载失败：{e}\n💡 检查网络/代理，或换 tiny 模型")

    def transcribe(self, audio_path):
        # 校验音频文件
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ 音频文件不存在：{audio_path}")

        # 关键3：自动转换音频为标准WAV（解决MP3解码崩溃）
        import tempfile
        import subprocess
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            # 转换为16kHz单声道WAV（faster-whisper原生支持，无解码冲突）
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10
            )
        except Exception as e:
            raise RuntimeError(f"❌ 需安装ffmpeg：\nMac: brew install ffmpeg\nLinux: apt install ffmpeg\n错误：{e}")

        # 执行转录（用标准WAV，避免崩溃）
        print(f"\n📼 正在转录音频：{audio_path}")
        segments, _ = self.model.transcribe(
            temp_wav,
            language="zh",
            beam_size=3,  # 降低计算量，避免内存溢出
            vad_filter=True,
            word_timestamps=False  # 关闭词级时间戳，减少计算
        )

        # 拼接结果 + 清理临时文件
        full_text = "".join([seg.text.strip() for seg in segments])
        os.unlink(temp_wav)

        return full_text

def main():
    if len(sys.argv) != 3:
        print("📚 纯在线模式使用：python whisper_online_fix_final.py <模型大小> <音频路径>")
        print("   示例：python whisper_online_fix_final.py tiny data/audio/test.mp3")
        sys.exit(1)

    model_size = sys.argv[1]
    audio_path = sys.argv[2]

    try:
        # 初始化在线模型（纯在线逻辑）
        whisper = OnlineWhisperFix(model_size=model_size)
        # 转录音频
        result = whisper.transcribe(audio_path)
        # 输出结果
        print("\n=================== 转录结果 ===================")
        print(result)
        print("===============================================")
        # 保存结果
        output_txt = os.path.splitext(audio_path)[0] + "_online_result.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n✅ 结果已保存到：{output_txt}")
    except Exception as e:
        print(f"\n❌ 运行失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()