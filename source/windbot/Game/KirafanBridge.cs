using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using YGOSharp.OCGWrapper;

namespace WindBot.Game
{
    /// <summary>
    /// Plays with the Kirafan learning AI: every game message goes to a Python process
    /// (kirafan_ai.bridge.server) that answers with the raw response bytes.
    /// Bots use it when their deck is "KirafanAI:&lt;name&gt;" or "KirafanAI:&lt;name&gt;@&lt;level&gt;" (difficulty
    /// 1-4, kirafan_ai.bridge.server LEVELS; see WindBot/KirafanAI/bridge.cfg); the model is
    /// WindBot/KirafanAI/models/&lt;name&gt;.npz, the bot's chat Dialogs/KirafanAI_&lt;level&gt;.json when there is one.
    /// </summary>
    public class KirafanBridge : IDisposable
    {
        public const string Prefix = "KirafanAI:";

        private readonly Process _proc;
        private readonly BinaryWriter _input;
        private readonly BinaryReader _output;

        public static bool IsBridgeDeck(string deck)
        {
            return deck != null && deck.StartsWith(Prefix);
        }

        public static string ModelName(string deck)
        {
            string name = deck.Substring(Prefix.Length);
            int at = name.IndexOf('@');
            return at < 0 ? name : name.Substring(0, at);
        }

        public static int Level(string deck)
        {
            int at = deck.IndexOf('@');
            int level;
            return at >= 0 && int.TryParse(deck.Substring(at + 1), out level) ? level : 0;
        }

        /// <summary>
        /// The bot's chat file: Dialogs/KirafanAI_S&lt;summoner&gt;.json (its deck's summoner card), else fallback.
        /// (kirafan_ai.bridge.server picks the same one for its lines.)
        /// </summary>
        public static string DialogName(int summoner, string fallback)
        {
            string name = "KirafanAI_S" + summoner;
            return summoner > 0 && File.Exists(Path.Combine(Path.Combine(Program.AssetPath, "Dialogs"), name + ".json"))
                ? name : fallback;
        }

        /// <summary>
        /// Deck for one duel: a random .ydk from Decks/KirafanAI_&lt;name&gt;_L&lt;level&gt;/ (the level's own decks), else
        /// from Decks/KirafanAI_&lt;name&gt;/ (written by kirafan_ai.bridge.register), else Decks/KirafanAI_&lt;name&gt;.ydk.
        /// </summary>
        public static string PickDeck(string name, int level = 0)
        {
            string decksDir = Path.Combine(Program.AssetPath, "Decks");
            foreach (string folder in new[] { level > 0 ? "KirafanAI_" + name + "_L" + level : null, "KirafanAI_" + name })
            {
                if (folder == null)
                    continue;
                string dir = Path.Combine(decksDir, folder);
                if (!Directory.Exists(dir))
                    continue;
                string[] decks = Directory.GetFiles(dir, "*.ydk");
                if (decks.Length > 0)
                    return Path.GetFullPath(decks[new Random().Next(decks.Length)]);
            }
            return "KirafanAI_" + name;
        }

        private static Dictionary<string, string> ReadConfig(string path)
        {
            var cfg = new Dictionary<string, string>();
            foreach (string line in File.ReadAllLines(path, Encoding.UTF8))
            {
                int eq = line.IndexOf('=');
                if (eq <= 0 || line.TrimStart().StartsWith("#"))
                    continue;
                cfg[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
            }
            return cfg;
        }

        [DataContract]
        private class SummonerChat
        {
            [DataMember]
            public string name { get; set; }
        }

        /// <summary>
        /// The bot's name in the room: "name" of Dialogs/KirafanAI_S&lt;summoner&gt;.json, else the summoner card's
        /// character (its card name without "(소환사 카드)"); EDOPro's "[AI] " prefix kept. current without a summoner.
        /// </summary>
        public static string PlayerName(int summoner, string current)
        {
            if (summoner <= 0)
                return current;
            string name = null;
            string path = Path.Combine(Path.Combine(Program.AssetPath, "Dialogs"), "KirafanAI_S" + summoner + ".json");
            try
            {
                if (File.Exists(path))
                    using (FileStream fs = File.OpenRead(path))
                        name = ((SummonerChat)new DataContractJsonSerializer(typeof(SummonerChat)).ReadObject(fs)).name;
            }
            catch (Exception)
            {
            }
            if (string.IsNullOrEmpty(name))
            {
                NamedCard card = NamedCard.Get(summoner);
                name = card == null ? null : card.Name.Replace("(소환사 카드)", "").Trim();
            }
            if (string.IsNullOrEmpty(name))
                return current;
            string full = (current != null && current.StartsWith("[AI] ") ? "[AI] " : "") + name;
            return full.Length > 20 ? full.Substring(0, 20) : full;
        }

        private readonly Action<string> _chat;

        public KirafanBridge(string name, int level, Action<string> chat)
        {
            _chat = chat;
            string root = Path.GetFullPath(Path.Combine(Program.AssetPath, "KirafanAI"));
            var cfg = ReadConfig(Path.Combine(root, "bridge.cfg"));
            // paths in bridge.cfg may be relative to WindBot/KirafanAI (portable bundle)
            string python = Path.GetFullPath(Path.Combine(root, cfg["python"]));
            string workdir = Path.GetFullPath(Path.Combine(root, cfg["workdir"]));
            string model = Path.Combine(Path.Combine(root, "models"), name);
            string logDir = Path.Combine(root, "logs");
            Directory.CreateDirectory(logDir);
            string log = Path.Combine(logDir, DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") + "_" + name + ".log");
            string mode = cfg.ContainsKey("mode") ? cfg["mode"] : "best";

            var psi = new ProcessStartInfo(python,
                "-u -m kirafan_ai.bridge.server --model \"" + model + "\" --mode " + mode +
                (level > 0 ? " --level " + level : "") + " --log \"" + log + "\"");
            psi.WorkingDirectory = workdir;
            psi.EnvironmentVariables["KIRAFAN_GAME_ROOT"] = Path.GetFullPath(Path.Combine(root, Path.Combine("..", "..")));
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardInput = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            _proc = Process.Start(psi);
            _proc.ErrorDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                    File.AppendAllText(log, e.Data + Environment.NewLine, Encoding.UTF8);
            };
            _proc.BeginErrorReadLine();
            _input = new BinaryWriter(_proc.StandardInput.BaseStream);
            _output = new BinaryReader(_proc.StandardOutput.BaseStream);
            Logger.WriteLine("KirafanAI bridge started: " + model);
        }

        /// <summary>
        /// Sends one game message (message id + body); returns the response, or null. Before it the bot may say
        /// lines in the chat: each comes as length -1, then [u32 size][UTF-8 text].
        /// </summary>
        public byte[] Handle(byte[] message)
        {
            _input.Write(message.Length);
            _input.Write(message);
            _input.Flush();
            int length = _output.ReadInt32();
            while (length == -1)
            {
                string text = Encoding.UTF8.GetString(_output.ReadBytes(_output.ReadInt32()));
                if (_chat != null)
                    _chat(text);
                length = _output.ReadInt32();
            }
            return length > 0 ? _output.ReadBytes(length) : null;
        }

        public void Dispose()
        {
            try
            {
                _input.Close();
                if (!_proc.WaitForExit(3000))
                    _proc.Kill();
            }
            catch (Exception)
            {
            }
        }
    }
}
