import numpy as np
import pretty_midi
from tqdm import tqdm
import sys
from chainer import dataset, cuda

from . import midi_to_score
from ..utils import get_array_in_fixed_size

import numpy as np
try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False
import os
import pkg_resources

from pretty_midi.containers import PitchBend
from pretty_midi.utilities import pitch_bend_to_semitones, note_number_to_hz

# DEFAULT_SF2 = 'TimGM6mb.sf2'

def get_error_message(sys_exc_info=None):
    ex, ms, tb = sys.exc_info() if sys_exc_info is None else sys_exc_info
    return '[Error]\n' + str(ex) + '\n' + str(ms)


class MyInstrument(pretty_midi.Instrument):
    # OVERRIDE
    def fluidsynth(self, sf2_path, fs=44100, rec_range=(None, None), num_channel=1):
        """Synthesize using fluidsynth.
        Parameters
        ----------
        fs : int
            Sampling rate to synthesize.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``pretty_midi``.
        rec_range : (rec_start, rec_end)
            rec_start : float / None
            rec_end : float / None
                specify recording interval.
        num_channel : 1 or 2
            monaural or stereo

        Returns
        -------
        synthesized : np.ndarray (shape==(wave_samples, num_channel), dtype==np.float)
            Waveform of the MIDI data, synthesized at ``fs``.
            scale order : 32768
        """
        # If sf2_path is None, use the included TimGM6mb.sf2 path
        # if sf2_path is None:
        #     sf2_path = pkg_resources.resource_filename(__name__, DEFAULT_SF2)

        if not _HAS_FLUIDSYNTH:
            raise ImportError("fluidsynth() was called but pyfluidsynth "
                              "is not installed.")

        if not os.path.exists(sf2_path):
            raise ValueError("No soundfont file found at the supplied path "
                             "{}".format(sf2_path))

        # If the instrument has no notes, return an empty array
        if len(self.notes) == 0:
            return np.array([])

        # Create fluidsynth instance
        fl = fluidsynth.Synth(samplerate=fs)
        # Load in the soundfont
        sfid = fl.sfload(sf2_path)
        # If this is a drum instrument, use channel 9 and bank 128
        if self.is_drum:
            channel = 9
            # Try to use the supplied program number
            res = fl.program_select(channel, sfid, 128, self.program)
            # If the result is -1, there's no preset with this program number
            if res == -1:
                # So use preset 0
                fl.program_select(channel, sfid, 128, 0)
        # Otherwise just use channel 0
        else:
            channel = 0
            fl.program_select(channel, sfid, 0, self.program)
        # Collect all notes in one list
        event_list = []
        for note in self.notes:
            event_list += [[note.start, 'note on', note.pitch, note.velocity]]
            event_list += [[note.end, 'note off', note.pitch]]
        for bend in self.pitch_bends:
            event_list += [[bend.time, 'pitch bend', bend.pitch]]
        for control_change in self.control_changes:
            event_list += [[control_change.time, 'control change',
                            control_change.number, control_change.value]]
        # Sort the event list by time, and secondarily by whether the event
        # is a note off
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        # Add some silence at the beginning according to the time of the first
        # event
        current_time = event_list[0][0]
        # Convert absolute seconds to relative samples
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        # Include 1 second of silence at the end
        event_list[-1][0] = 1.

        # total sample
        total_time = current_time + np.sum([e[0] for e in event_list])
        total_sample = int(np.ceil(fs * total_time))

        # rec range
        rec_start_sec, rec_end_sec = rec_range
        rec_start_sample = 0 if rec_start_sec is None else int(round(fs * rec_start_sec))
        rec_end_sample = total_sample if rec_end_sec is None else int(round(fs * rec_end_sec))

        # Pre-allocate output array
        synthesized = np.zeros([rec_end_sample - rec_start_sample, num_channel], dtype=np.float32)

        # Iterate over all events
        for event in event_list:
            # 欲しい時刻に達しているかどうか
            rec_started = True if rec_start_sec is None else rec_start_sec <= current_time
            rec_ended = False if rec_end_sec is None else rec_end_sec <= current_time

            # Process events based on type
            if event[1] == 'note on' and rec_started:
                fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off' and rec_started:
                fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                fl.cc(channel, event[2], event[3])
            # Add in these samples
            current_sample = int(fs*current_time)
            end = int(fs*(current_time + event[0]))

            # 欲しい時刻に達するまでは，波形の生成をサボる．
            if not rec_started:
                current_time += event[0]
                continue
            if rec_ended:
                break

            samples = fl.get_samples(end - current_sample).reshape(-1, 2)
            # NOTE: get_samples の返り値は dtype==np.int16 です．
            if num_channel == 1:
                samples = np.mean(samples.astype(np.float32), axis=1)
            synthesized[current_sample-rec_start_sample:end-rec_start_sample] += samples[:len(synthesized) - (current_sample-rec_start_sample)]

            # Increment the current sample
            current_time += event[0]
        # Close fluidsynth
        fl.delete()

        return synthesized

    # OVERRIDE
    def get_piano_roll(self, fs=100, times=None,
                       pedal_threshold=64, rec_range=(None, None), mode='hold'):
        """Compute a piano roll matrix of this instrument.
        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./fs`` seconds.
        times : np.ndarray
            Times of the start of each column in the piano roll.
            Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.
        Returns
        -------
        piano_roll : np.ndarray, shape=(128,times.shape[0])
            Piano roll of this instrument.
        """
        # rec_range と機能が衝突して紛らわしいので times は使用不可にしておく．
        assert times is None, '`times` should be None. If you want to restrict time range, use `rec_range` argument.'

        # TODO: この実装では最初に曲全体のデータを用意するが，これは効率が悪いので，なんとかしたい．
        if rec_range != (None, None):
            rec_start_sec = 0 if rec_range[0] is None else rec_range[0]
            rec_end_sec = self.get_end_time() if rec_range[1] is None else rec_range[1]

            rec_start_sample = int(round(rec_start_sec * fs))
            rec_end_sample = int(round(rec_end_sec * fs))
            times = np.arange(rec_start_sample, rec_end_sample) / fs
            # この times は正確に 1/fs の倍数にしておく必要がある．さもなくば，スムージングが発動してしまう．            
        # TODO: 無駄に高機能な times を使って実装しているので，最後の部分で，本来なら一括コピーで済む配列操作が，1列ずつコピーされ，遅くなるはず．

        # If there are no notes, return an empty matrix
        if self.notes == []:
            return np.array([[]]*128)
        # Get the end time of the last event
        end_time = self.get_end_time()
        # Extend end time if one was provided
        if times is not None and times[-1] > end_time:
            end_time = times[-1]
        # Allocate a matrix of zeros - we will add in as we go
        piano_roll = np.zeros((128, int(round(fs*end_time))+1))
        # Drum tracks don't have pitch, so return a matrix of zeros
        if self.is_drum:
            if times is None:
                return piano_roll
            else:
                return np.zeros((128, times.shape[0]))
        # Add up piano roll matrix, note-by-note
        if mode == 'hold':
            for note in self.notes:
                piano_roll[note.pitch, int(note.start*fs):int(note.end*fs)] += note.velocity
        elif mode == 'onset':
            for note in self.notes:
                piano_roll[note.pitch, int(note.start*fs)] += note.velocity
        else:   
            raise ValueError      

        if mode == 'hold':
            # Process sustain pedals
            if pedal_threshold is not None:
                CC_SUSTAIN_PEDAL = 64
                time_pedal_on = 0
                is_pedal_on = False
                for cc in [_e for _e in self.control_changes
                           if _e.number == CC_SUSTAIN_PEDAL]:
                    time_now = int(cc.time*fs)
                    is_current_pedal_on = (cc.value >= pedal_threshold)
                    if not is_pedal_on and is_current_pedal_on:
                        time_pedal_on = time_now
                        is_pedal_on = True
                    elif is_pedal_on and not is_current_pedal_on:
                        # For each pitch, a sustain pedal "retains"
                        # the maximum velocity up to now due to
                        # logarithmic nature of human loudness perception
                        subpr = piano_roll[:, time_pedal_on:time_now]

                        # Take the running maximum
                        pedaled = np.maximum.accumulate(subpr, axis=1)
                        piano_roll[:, time_pedal_on:time_now] = pedaled
                        is_pedal_on = False

        # Process pitch changes
        # Need to sort the pitch bend list for the following to work
        ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
        # Add in a bend of 0 at the end of time
        end_bend = PitchBend(0, end_time)
        for start_bend, end_bend in zip(ordered_bends,
                                        ordered_bends[1:] + [end_bend]):
            # Piano roll is already generated with everything bend = 0
            if np.abs(start_bend.pitch) < 1:
                continue
            # Get integer and decimal part of bend amount
            start_pitch = pitch_bend_to_semitones(start_bend.pitch)
            bend_int = int(np.sign(start_pitch)*np.floor(np.abs(start_pitch)))
            bend_decimal = np.abs(start_pitch - bend_int)
            # Column indices effected by the bend
            bend_range = np.r_[int(start_bend.time*fs):int(end_bend.time*fs)]
            # Construct the bent part of the piano roll
            bent_roll = np.zeros(piano_roll[:, bend_range].shape)
            # Easiest to process differently depending on bend sign
            if start_bend.pitch >= 0:
                # First, pitch shift by the int amount
                if bend_int is not 0:
                    bent_roll[bend_int:] = piano_roll[:-bend_int, bend_range]
                else:
                    bent_roll = piano_roll[:, bend_range]
                # Now, linear interpolate by the decimal place
                bent_roll[1:] = ((1 - bend_decimal)*bent_roll[1:] +
                                 bend_decimal*bent_roll[:-1])
            else:
                # Same procedure as for positive bends
                if bend_int is not 0:
                    bent_roll[:bend_int] = piano_roll[-bend_int:, bend_range]
                else:
                    bent_roll = piano_roll[:, bend_range]
                bent_roll[:-1] = ((1 - bend_decimal)*bent_roll[:-1] +
                                  bend_decimal*bent_roll[1:])
            # Store bent portion back in piano roll
            piano_roll[:, bend_range] = bent_roll

        if times is None:
            return piano_roll
        """
        piano_roll_integrated = np.zeros((128, times.shape[0]))
        # Convert to column indices
        times = np.array(np.round(times*fs), dtype=np.int)
        for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end],
                                                  axis=1)
        """
        # 今回は，times は連続する整数列であることがわかっているので，
        piano_roll_integrated = get_array_in_fixed_size(
            piano_roll,
            axis=1,
            size_dest=len(times),
            start_src=int(np.round(times[0]*fs))
        )

        return piano_roll_integrated


class MidiSampler(object):
    def __init__(self, midi_file):
        self.midi_file = midi_file
        self.midi_data = pretty_midi.PrettyMIDI(midi_file)
        # cast class (Instrument => MyInstrument)
        for inst in self.midi_data.instruments:
            inst.__class__ = MyInstrument

    def sample(self, length_sec, wave_sr=44100, score_sr=44100/512, allow_drum=True, num_part=None, sf2_path=None, program=None, num_channel=1, verbose=False):
        """
        指定された秒数のデータをランダムにサンプリングして返す．
        ------
        Args:
            length_sec (float):
                欲しい秒数．長すぎる場合は曲全体が返る．
            wave_sr (float):
                wave のサンプリングレート．
            score_sr (float):
                score のサンプリングレート．
            allow_drum (bool):
                ドラムパートを許容するか否か．
            num_part (None / int):
                指定されたパート数だけランダムに選んで返します．
                None または多すぎる場合は，返りうる全パートを返します．
            program (None / list of int)
                許容する MIDI 音色プログラム番号(0-127)のリスト．
                None の場合は，全て許容されます．
            num_channel (1 or 2)
                wave のモノラル or ステレオ

        Returns:
            wave:
                shape == (パート数, wave_samples, num_channel)
                値のスケールは ±32768 程度（この範囲内であることは保証されない）．
            score:
                shape == (パート数, ノートナンバー, score_samples)
                ノートオンからノートオフまでが存在する箇所にベロシティ値(0-127)が加算された配列です
            score_onset:
                shape == (パート数, ノートナンバー, score_samples)
                ノートオンの箇所にベロシティ値(0-127)が加算された配列です

            ※ wave_samples と score_samples は次式により計算されます：
                wave_samples = int(round(length_sec * wave_sr))
                score_samples = int(round(length_sec * score_sr))

        Examples:
            ```
            import numpy as np
            import matplotlib.pyplot as plt
            import mimicopynet as mcn
            ms = mcn.data.MidiSampler('hoge.mid')
            wave, score, onset = ms.sample(3.0, num_part=2, allow_drum=True)

            # listen to sound
            mcn.ipython_utils.show_audio(np.sum(wave, axis=0), 44100)

            # visualize onset
            plt.imshow(np.sum(onset, axis=0).astype(np.bool), origin='lower')
            plt.show()
            ```
        """
        import time
        ts = []
        ts.append(time.time())

        # サンプル数の計算．
        wave_samples = int(round(length_sec * wave_sr))
        score_samples = int(round(length_sec * score_sr))

        # 取得位置の決定（秒数）
        total_length_sec = self.midi_data.get_end_time()
        start_sec = np.random.rand() * (total_length_sec + length_sec) - length_sec
        end_sec = start_sec + length_sec

        # 取得位置の決定（サンプル数）
        start_wave_sample = int(np.floor(start_sec * wave_sr))
        end_wave_sample = start_wave_sample + wave_samples
        start_score_sample = int(np.floor(start_sec * score_sr))
        end_score_sample = start_score_sample + score_samples

        # 許容されるパートを調べる．
        is_possible = np.ones(len(self.midi_data.instruments), dtype=np.bool)
        if not allow_drum:
            is_drum = np.array([inst.is_drum for inst in self.midi_data.instruments], dtype=np.bool)
            is_possible = np.logical_and(is_possible, np.logical_not(is_drum))
        if program is not None:
            is_allowed_program = np.array([inst.program in program for inst in self.midi_data.instruments], dtype=np.bool)
            is_possible = np.logical_and(is_possible, is_allowed_program)

        # パートのランダム選択．
        possible_parts = np.flatnonzero(is_possible)
        if num_part is None:
            parts = possible_parts
        else:
            num_part = min(num_part, len(possible_parts))
            parts = np.sort(np.random.choice(possible_parts, num_part, replace=False))

        ts.append(time.time())

        # 少しマージンをつけて録音をする（「最初から鳴っている音」を収録するため）
        pre_margin_samples = int(round(wave_sr * 1.0))
        post_margin_samples = int(round(wave_sr * 0.01))
        rec_range = ((start_wave_sample - pre_margin_samples)/wave_sr, (end_wave_sample + post_margin_samples)/wave_sr)

        # 録音
        wave = np.zeros([len(parts), wave_samples, num_channel], dtype=np.float32)
        for i, p in enumerate(parts):
            inst = self.midi_data.instruments[p]
            wave_with_margin = inst.fluidsynth(fs=wave_sr, sf2_path=sf2_path, rec_range=rec_range, num_channel=num_channel)
            # wave からマージンを取り除き， (start_sec, end_sec) に相当する部分を切り出す．
            wave[i] = wave_with_margin[pre_margin_samples:pre_margin_samples+wave_samples]

        ts.append(time.time())

        # スコア情報の取得
        rec_range = (start_score_sample/score_sr, end_score_sample/score_sr)
        score_hold = np.zeros([len(parts), 128, score_samples], dtype=np.float32)
        score_onset = np.zeros([len(parts), 128, score_samples], dtype=np.float32)
        for i, p in enumerate(parts):
            inst = self.midi_data.instruments[p]
            score_hold[i] = inst.get_piano_roll(fs=score_sr, rec_range=rec_range)
            score_onset[i] = inst.get_piano_roll(fs=score_sr, rec_range=rec_range, mode='onset')

        ts.append(time.time())

        ts = np.array(ts)

        if verbose:
            print('position (sec): {:.1f}-{:.1f} (total: {:.1f})'.format(
                start_sec, end_sec, total_length_sec))
            print('parts: {} (total: {})'.format(parts, len(self.midi_data.instruments)))
            for p in parts:
                print('    [{}] program: {}, name: {}'.format(p, self.midi_data.instruments[p].program, self.midi_data.instruments[p].name))
            print('synthesizing wave: {:.3f} sec'.format(ts[2] - ts[1]))
            print('fetching score: {:.3f} sec'.format(ts[3] - ts[2]))

        return wave, score_hold, score_onset



class MidiDataset(dataset.DatasetMixin):
    def __init__(self, midis, size, sf2_path, length_sec=1.0, num_part=3, allow_drum=False, program=None, num_channel=1, ):
        """
        MIDIファイルから直接データを生成する．
        注意： この Dataset を SerialIterator に渡すとき，shuffle=False にしてください．
              True にするのは，意味がないばかりか，時間をとても無題にする可能性があります．

        TODO どちらかというと，Dataset ではなく Iterator を作った方が良さそう？？？
        --------
        Args:
            midis (iterable of str):
                データのランダム抽出元となる，
                WSH5形式のファイルへのパスを表す文字列の iterable
            size (int):
                データセットのサイズ．これは「1エポック」の件数を決めるためだけに用いられる．
            length_sec (float / callable):
                1つのデータに含まれる秒数．定数で与えるか，レベル(int)を引数に float 値を返す callable を指定．
            allow_drum (bool / callable):
                ドラムパートを許容するか否か．定数で与えるか，レベル(int)を引数に bool 値を返す callable を指定．
            num_part (None / int / callable):
                マージするパート数．定数で与えるか，レベル(int)を引数に int 値を返す callable を指定．
                None の場合は，選択されうる全パートがマージされます．
            program (None / list of int / callable)
                許容する MIDI 音色プログラム番号(0-127)のリスト．
                None の場合は，全て許容されます．
            num_channel (1 or 2)
                wave のモノラル or ステレオ
        """
        self.midi_samplers = []
        for m in tqdm(list(midis)):
            try:
                self.midi_samplers.append(MidiSampler(m))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                print('[On loading {}]'.format(m))
                print(get_error_message())
                """
                NOTE: これまでに出たことのあるエラー一覧
                <class 'KeyError'>
                    (-1, 255)
                    (-4, 255)
                    (16, 1)
                    (-5, 255)
                    (-2, 255)
                <class 'OSError'>
                    data byte must be in range 0..127
                <class 'OSError'>
                    MThd not found. Probably not a MIDI file
                <class 'EOFError'>

                <class 'IndexError'>
                    list index out of range
                <class 'OSError'>
                    running status without last_status
                <class 'OSError'>
                    no MTrk header at start of track
                <class 'ValueError'>
                    MIDI file has a largest tick of 42949772801, it is likely corrupt
                """

        self.size = size
        self.length_sec = length_sec
        self.num_part = num_part
        self.sf2_path = sf2_path
        self.allow_drum = allow_drum
        self.program = program
        self.num_channel = num_channel
        self.level = 0

        # 統計情報を持っておきたい
        self.stats_wave_amp = []

    def __len__(self):
        return self.size

    def get_example(self, i, verbose=False):
        """
        ランダムにデータを返す．
        ----
        Args:
            i (int): インデクス（使われない）
        Returns:
            以下の tuple
                wave (np.ndarray):
                    波形情報．shape==(samples_of_wave, num_channel) dtype==np.float32
                    スケールは ±1 程度（この範囲に収まっていることは保証されない）
                score_feats (np.ndarray):
                    スコア情報．shape==(2, 128, samples) dtype==np.int32
                    score_feats[0] は hold 譜面．(ノートオンからオフまで 1 が入る．それ以外は 0)
                    score_feats[1] は onset 譜面．(ノートオンの瞬間だけ 1 が入る．それ以外は 0)

            （タスク設計は，loss関数の側で行ってください．）

        """
        ms = np.random.choice(self.midi_samplers)
        length_sec = self.length_sec(self.level) if callable(self.length_sec) else self.length_sec
        allow_drum = self.allow_drum(self.level) if callable(self.allow_drum) else self.allow_drum
        num_part = self.num_part(self.level) if callable(self.num_part) else self.num_part
        wave, score, score_onset = ms.sample(
            length_sec,
            sf2_path=self.sf2_path,
            allow_drum=allow_drum,
            num_part=num_part,
            program=self.program,
            num_channel=self.num_channel,
            verbose=verbose
        )
        # TODO
        # 混ぜ合わせを 1:1 からずらしたり，逆位相にしたり，オフセット少しずらしたり，
        # ホワイトノイズなど加えたり，ピッチシフトしたり，など色々なオーグメンテーションができる．

        # パートのマージ．
        wave = np.sum(wave, axis=0)  # (wave_samples, num_channel)
        score = np.sum(score, axis=0)  # (pitch, score_samples)
        score_onset = np.max(score_onset, axis=0)  # (pitch, score_samples)
        score_feats = np.stack([score, score_onset])  # (2, pitch, score_samples)
        # NOTE: sum の代わりに mean 数を使うと，パート数の情報がリークしてしまうと考えられる？
        #       (0パートの時に nan が出るという問題もある．)

        # wave の正規化
        wave /= 32768
        # NOTE: 最大絶対値での正規化は「無音＋微小ノイズ」も増幅してしまうので避ける．
        # maxi = np.max(np.abs(wave))
        # if maxi > 0:
        #     wave /= maxi

        # score のバイナリ化．
        score_feats = score_feats.astype(np.bool).astype(np.int32)

        # 統計情報の更新
        self.stats_wave_amp.append(np.max(np.abs(wave)))

        return wave, score_feats

